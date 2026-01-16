"""
KV-cached rollout engine for rollout-light v2 scoring.

Uses paired sampling, event-based answer probability metrics,
and receiver-side masking for screening.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import re
import logging
from datetime import datetime

import torch
import torch.nn.functional as F

logger = logging.getLogger("edgepatch")


def _ts() -> str:
    """Return current timestamp string for logging."""
    return datetime.now().strftime("%H:%M:%S")


# Answer detection patterns
ANSWER_PATTERNS = [
    r'\\boxed\{([^}]*)\}',     # LaTeX boxed
    r'Answer:\s*(.+?)(?:\n|$)', # "Answer: X"
    r'=\s*(\d+(?:\.\d+)?)\s*$', # "= 42" at end
    r'The answer is\s*(.+?)(?:\.|$)',
]


@dataclass  
class RolloutResult:
    """Result of running rollouts at a decision point for a chunk."""
    decision_point_idx: int
    chunk_idx: int
    baseline_reached_answer: bool
    intervened_reached_answer: bool
    answer_prob_shift: float           # |P(answer | base) - P(answer | inter)|
    answer_content_divergence: float   # Did answers differ?
    token_overlap: float               # Fallback metric
    first_token_kl: float             # KL between first generated tokens
    
    def to_dict(self) -> dict:
        return {
            "decision_point_idx": self.decision_point_idx,
            "chunk_idx": self.chunk_idx,
            "answer_prob_shift": round(self.answer_prob_shift, 4),
            "answer_content_divergence": round(self.answer_content_divergence, 4),
            "baseline_reached_answer": self.baseline_reached_answer,
            "intervened_reached_answer": self.intervened_reached_answer,
            "first_token_kl": round(self.first_token_kl, 4),
        }



def ensure_dynamic_cache(past_key_values):
    """
    Ensure KV cache is a DynamicCache object, converting from tuple if needed.
    
    Workaround for Llama models that crash with 'tuple object has no attribute get_seq_length'.
    """
    from transformers.cache_utils import DynamicCache
    
    # Already a cache object?
    if hasattr(past_key_values, 'get_seq_length'):
        return past_key_values
        
    # Check if it's a tuple (legacy format)
    if isinstance(past_key_values, tuple):
        try:
            cache = DynamicCache()
            for layer_idx, layer_kv in enumerate(past_key_values):
                key, value = layer_kv
                cache.update(key, value, layer_idx)
            return cache
        except Exception as e:
            print(f"[{_ts()}] Warning: Failed to convert tuple to DynamicCache: {e}", flush=True)
            return past_key_values
            
    return past_key_values


def get_prefix_kv_cache(
    model,
    input_ids: torch.Tensor,
    up_to_position: int,
) -> tuple[Any, torch.Tensor]:
    """
    Compute and cache KV states up to a given position.
    
    Args:
        model: The language model  
        input_ids: Full input token IDs [1, seq_len]
        up_to_position: Cache KV up to this position (exclusive)
        
    Returns:
        past_key_values: Cached KV states (DynamicCache or tuple)
        prefix_ids: Input IDs up to position
    """
    with torch.no_grad():
        prefix_ids = input_ids[:, :up_to_position]
        outputs = model(prefix_ids, use_cache=True)
        # Ensure we return a DynamicCache if possible, as Llama expects it
        pkv = ensure_dynamic_cache(outputs.past_key_values)
        return pkv, prefix_ids


def ablate_kv_for_span(
    past_key_values,
    span_start: int,
    span_end: int,
):
    """
    Zero out KV cache entries for a token span (KV ablation).
    
    Works with both DynamicCache and legacy tuple formats.
    
    Args:
        past_key_values: DynamicCache or tuple of (key, value) tensors per layer
        span_start: Start token index to ablate
        span_end: End token index to ablate (exclusive)
        
    Returns:
        Modified past_key_values with ablated span (same type as input)
    """
    from transformers.cache_utils import DynamicCache
    
    # Check if it's a DynamicCache
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        # DynamicCache - create a new one with ablated values
        new_cache = DynamicCache()
        
        for layer_idx in range(len(past_key_values.key_cache)):
            key = past_key_values.key_cache[layer_idx].clone()
            value = past_key_values.value_cache[layer_idx].clone()
            
            # Key and Value shape: [batch, n_heads, seq_len, head_dim]
            seq_len = key.shape[2]
            if span_start < seq_len and span_end <= seq_len:
                key[:, :, span_start:span_end, :] = 0.0
                value[:, :, span_start:span_end, :] = 0.0
            
            # Add to new cache using the update method
            new_cache.update(key, value, layer_idx)
        
        return new_cache
    else:
        # Legacy tuple format
        ablated = []
        for layer_kv in past_key_values:
            key, value = layer_kv
            # Key and Value shape: [batch, n_heads, seq_len, head_dim]
            
            # Create copies
            key_ablated = key.clone()
            value_ablated = value.clone()
            
            # Zero out the span
            seq_len = key.shape[2]
            if span_start < seq_len and span_end <= seq_len:
                key_ablated[:, :, span_start:span_end, :] = 0.0
                value_ablated[:, :, span_start:span_end, :] = 0.0
            
            ablated.append((key_ablated, value_ablated))
        
        return tuple(ablated)


def generate_continuation_paired(
    model,
    tokenizer,
    prefix_ids: torch.Tensor,
    baseline_kv: Any,
    intervened_kv: Any,
    horizon: int,
    temperature: float = 0.7,
    seed: int = 42,
) -> tuple[str, str, list[int], list[int], float]:
    """
    Generate paired continuations with same RNG seed.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for decoding
        prefix_ids: Token IDs of the prefix [1, prefix_len]
        baseline_kv: KV cache for baseline
        intervened_kv: KV cache for intervened run
        horizon: Number of tokens to generate
        temperature: Sampling temperature
        seed: Random seed for reproducibility (same for both)
        
    Returns:
        baseline_text, intervened_text, baseline_tokens, intervened_tokens, first_token_kl
    """
    device = prefix_ids.device
    
    # Generate baseline
    torch.manual_seed(seed)
    baseline_tokens = []
    current_kv = baseline_kv
    current_token = prefix_ids[:, -1:]
    baseline_first_logits = None
    
    with torch.no_grad():
        for step in range(horizon):
            outputs = model(current_token, past_key_values=current_kv, use_cache=True)
            logits = outputs.logits[:, -1, :]
            current_kv = outputs.past_key_values
            
            if step == 0:
                baseline_first_logits = logits.clone()
            
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            baseline_tokens.append(next_token.item())
            current_token = next_token
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Generate intervened with SAME seed
    torch.manual_seed(seed)
    intervened_tokens = []
    current_kv = intervened_kv
    current_token = prefix_ids[:, -1:]
    intervened_first_logits = None
    
    with torch.no_grad():
        for step in range(horizon):
            outputs = model(current_token, past_key_values=current_kv, use_cache=True)
            logits = outputs.logits[:, -1, :]
            current_kv = outputs.past_key_values
            
            if step == 0:
                intervened_first_logits = logits.clone()
            
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            intervened_tokens.append(next_token.item())
            current_token = next_token
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Compute first-token KL divergence
    first_token_kl = 0.0
    if baseline_first_logits is not None and intervened_first_logits is not None:
        base_logp = F.log_softmax(baseline_first_logits.float(), dim=-1)
        inter_logp = F.log_softmax(intervened_first_logits.float(), dim=-1)
        base_p = torch.exp(base_logp)
        first_token_kl = torch.sum(base_p * (base_logp - inter_logp)).item()
        first_token_kl = max(0.0, first_token_kl)
    
    baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=True)
    intervened_text = tokenizer.decode(intervened_tokens, skip_special_tokens=True)
    
    return baseline_text, intervened_text, baseline_tokens, intervened_tokens, first_token_kl


def detect_answer(text: str) -> tuple[bool, Optional[str]]:
    """
    Detect if text contains an answer pattern.
    
    Returns:
        (found_answer, answer_content)
    """
    for pattern in ANSWER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return True, match.group(1) if match.groups() else match.group(0)
    return False, None


def compute_answer_probability_shift(
    baseline_text: str,
    intervened_text: str,
    baseline_reached: bool,
    intervened_reached: bool,
    baseline_answer: Optional[str],
    intervened_answer: Optional[str],
) -> tuple[float, float]:
    """
    Compute event-based answer probability shift.
    
    Returns:
        answer_prob_shift: Difference in P(reaching answer)
        answer_content_divergence: Did the answers differ?
    """
    # Simple binary: did we reach an answer?
    base_prob = 1.0 if baseline_reached else 0.0
    inter_prob = 1.0 if intervened_reached else 0.0
    
    answer_prob_shift = abs(base_prob - inter_prob)
    
    # Content divergence: if both reached answer, did they differ?
    if baseline_reached and intervened_reached:
        if baseline_answer and intervened_answer:
            # Normalize and compare
            base_norm = re.sub(r'\s+', '', baseline_answer.lower())
            inter_norm = re.sub(r'\s+', '', intervened_answer.lower())
            answer_content_divergence = 0.0 if base_norm == inter_norm else 1.0
        else:
            answer_content_divergence = 0.0
    elif baseline_reached != intervened_reached:
        # One reached, other didn't - high divergence
        answer_content_divergence = 1.0
    else:
        # Neither reached - check for partial divergence via text difference
        answer_content_divergence = 0.0
    
    return answer_prob_shift, answer_content_divergence


def run_paired_rollouts(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    decision_point_token_idx: int,
    chunk_span_start: int,
    chunk_span_end: int,
    chunk_idx: int,
    rollout_k: int = 4,
    rollout_h: int = 64,
    temperature: float = 0.7,
) -> RolloutResult:
    """
    Run K paired rollouts at a decision point with and without chunk ablation.
    
    Uses the same RNG seeds for baseline and intervened to enable fair comparison.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        input_ids: Full input token IDs [1, seq_len]
        decision_point_token_idx: Token index of decision point
        chunk_span_start: Start of chunk to ablate
        chunk_span_end: End of chunk to ablate
        chunk_idx: Index of this chunk
        rollout_k: Number of paired rollouts
        rollout_h: Horizon (tokens) per rollout
        temperature: Sampling temperature
        
    Returns:
        RolloutResult with aggregated metrics
    """
    # Get prefix KV cache up to decision point
    baseline_kv, prefix_ids = get_prefix_kv_cache(model, input_ids, decision_point_token_idx)
    
    # Create ablated version of KV cache
    ablated_kv = ablate_kv_for_span(baseline_kv, chunk_span_start, chunk_span_end)
    
    baseline_reached_count = 0
    intervened_reached_count = 0
    total_prob_shift = 0.0
    total_content_div = 0.0
    total_overlap = 0.0
    total_first_kl = 0.0
    
    for k in range(rollout_k):
        seed = 42 + k  # Different but reproducible seeds per rollout pair
        
        # Generate paired continuations
        base_text, inter_text, base_tokens, inter_tokens, first_kl = generate_continuation_paired(
            model, tokenizer, prefix_ids,
            baseline_kv, ablated_kv,
            horizon=rollout_h, temperature=temperature, seed=seed
        )
        
        # Detect answers
        base_reached, base_answer = detect_answer(base_text)
        inter_reached, inter_answer = detect_answer(inter_text)
        
        if base_reached:
            baseline_reached_count += 1
        if inter_reached:
            intervened_reached_count += 1
        
        # Compute answer probability shift
        prob_shift, content_div = compute_answer_probability_shift(
            base_text, inter_text, base_reached, inter_reached, base_answer, inter_answer
        )
        total_prob_shift += prob_shift
        total_content_div += content_div
        total_first_kl += first_kl
        
        # Token overlap (fallback metric)
        if base_tokens and inter_tokens:
            base_set = set(base_tokens)
            inter_set = set(inter_tokens)
            if base_set or inter_set:
                overlap = len(base_set & inter_set) / len(base_set | inter_set)
            else:
                overlap = 1.0
            total_overlap += overlap
        else:
            total_overlap += 1.0
    
    # Aggregate across K rollouts
    return RolloutResult(
        decision_point_idx=decision_point_token_idx,
        chunk_idx=chunk_idx,
        baseline_reached_answer=(baseline_reached_count > rollout_k / 2),
        intervened_reached_answer=(intervened_reached_count > rollout_k / 2),
        answer_prob_shift=total_prob_shift / rollout_k,
        answer_content_divergence=total_content_div / rollout_k,
        token_overlap=total_overlap / rollout_k,
        first_token_kl=total_first_kl / rollout_k,
    )


def run_screening_with_receiver_masking(
    model,
    input_ids: torch.Tensor,
    decision_point: 'DecisionPoint',
    chunk_spans: list,
    baseline_log_probs: torch.Tensor,
    config,
    model_info: dict,  # Now passed from caller
) -> list[tuple[int, float]]:
    """
    Run receiver-side masking to screen chunks at a decision point.
    
    Tests: "did this chunk influence the branch decision?"
    
    Args:
        model: The language model
        input_ids: Token IDs [1, seq_len]
        decision_point: DecisionPoint object with top_chunks
        chunk_spans: List of TokenSpan objects
        baseline_log_probs: Baseline log-probs [seq_len, vocab]
        config: EdgePatchConfig
        model_info: Dict with num_layers, num_heads, layer_to_module
        
    Returns:
        List of (chunk_idx, kl_score) sorted by KL
    """
    from edgepatch.masking import ScopedAttentionMasker
    
    device = input_ids.device
    dp_idx = decision_point.token_idx
    
    # Get baseline distribution at decision point
    if dp_idx > 0 and dp_idx - 1 < baseline_log_probs.shape[0]:
        base_log_probs = baseline_log_probs[dp_idx - 1].to(device)
    else:
        return [(c, 0.0) for c in decision_point.top_chunks]
    
    chunk_kl_scores = []
    
    # Only evaluate top-L chunks (already screened by attention)
    for chunk_idx in decision_point.top_chunks:
        if chunk_idx >= len(chunk_spans):
            continue
            
        chunk_span = chunk_spans[chunk_idx]
        
        # Skip chunks after decision point (no causal influence)
        if chunk_span.start_token >= dp_idx:
            continue
        
        k_positions = list(range(chunk_span.start_token, chunk_span.end_token))
        q_positions = [dp_idx]  # Receiver-side: decision token cannot attend to chunk
        
        with ScopedAttentionMasker(
            model,
            model_info,
            config.edge_layers,
            config.edge_heads,
            validate_on_exit=False,
        ) as masker:
            masker.set_mask_positions(q_positions, k_positions)
            
            with torch.no_grad():
                outputs = model(input_ids, use_cache=False)
                logits = outputs.logits
            
            # KL at decision point
            masked_log_probs = F.log_softmax(logits[0, dp_idx - 1, :].float(), dim=-1)
            p_base = torch.exp(base_log_probs)
            kl = torch.sum(p_base * (base_log_probs - masked_log_probs)).item()
            kl = max(0.0, kl)
        
        chunk_kl_scores.append((chunk_idx, kl))
    
    # Sort by KL (higher = more important)
    chunk_kl_scores.sort(key=lambda x: x[1], reverse=True)
    
    return chunk_kl_scores
