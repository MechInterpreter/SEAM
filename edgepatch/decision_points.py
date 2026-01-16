"""
Decision point discovery for rollout-light v2 scoring.

Model-driven discovery via entropy/margin curves.
NO full attention extraction - uses cheap heuristic screening.
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


@dataclass
class DecisionPoint:
    """A potential branch point in the reasoning trace."""
    token_idx: int           # Token index in the sequence
    entropy: float           # Token-level entropy at this position
    margin: float            # Top-1 vs Top-2 probability margin
    is_sentence_boundary: bool = False
    is_answer_token: bool = False
    top_chunks: list[int] = field(default_factory=list)  # Top-L chunk indices for screening
    
    def to_dict(self) -> dict:
        return {
            "token_idx": self.token_idx,
            "entropy": round(self.entropy, 4),
            "margin": round(self.margin, 4),
            "is_sentence_boundary": self.is_sentence_boundary,
            "is_answer_token": self.is_answer_token,
            "top_chunks": self.top_chunks,
        }


def compute_entropy_and_margin_curves(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute token-level entropy and margin curves from logits.
    
    Args:
        logits: [seq_len, vocab_size] tensor of logits
        
    Returns:
        entropy: [seq_len] tensor of entropy values
        margin: [seq_len] tensor of top-1 vs top-2 margins
    """
    # Convert to probabilities
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    
    # Entropy: -sum(p * log p)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # Margin: p(top1) - p(top2)
    top2_probs, _ = torch.topk(probs, k=2, dim=-1)
    margin = top2_probs[:, 0] - top2_probs[:, 1]
    
    return entropy, margin


def find_local_maxima(values: torch.Tensor, window: int = 5) -> list[int]:
    """Find indices of local maxima in a 1D tensor."""
    maxima = []
    values_np = values.cpu().numpy()
    
    for i in range(window, len(values_np) - window):
        window_vals = values_np[i - window:i + window + 1]
        if values_np[i] == window_vals.max():
            maxima.append(i)
    
    return maxima


def find_local_minima(values: torch.Tensor, window: int = 5) -> list[int]:
    """Find indices of local minima (sharp drops) in a 1D tensor."""
    minima = []
    values_np = values.cpu().numpy()
    
    for i in range(window, len(values_np) - window):
        window_vals = values_np[i - window:i + window + 1]
        if values_np[i] == window_vals.min():
            minima.append(i)
    
    return minima


def find_sentence_boundaries(
    input_ids: torch.Tensor,
    tokenizer,
) -> list[int]:
    """Find token indices that correspond to sentence boundaries."""
    boundaries = []
    
    # Decode and find period/question/exclamation positions
    text = tokenizer.decode(input_ids[0])
    
    # Get offset mapping if available
    try:
        encoding = tokenizer(text, return_offsets_mapping=True)
        offset_mapping = encoding.offset_mapping
    except:
        return []
    
    for match in re.finditer(r'[.!?]\s+', text):
        char_idx = match.end()
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= char_idx < end:
                boundaries.append(idx)
                break
    
    return boundaries


def find_answer_tokens(
    input_ids: torch.Tensor,
    tokenizer,
) -> list[int]:
    """Find token indices near 'Answer:' or boxed{} patterns."""
    answer_tokens = []
    text = tokenizer.decode(input_ids[0])
    
    patterns = [r'Answer:\s*', r'\\boxed\{', r'Therefore,?\s+the\s+answer']
    
    try:
        encoding = tokenizer(text, return_offsets_mapping=True)
        offset_mapping = encoding.offset_mapping
    except:
        return []
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            char_idx = match.start()
            for idx, (start, end) in enumerate(offset_mapping):
                if start <= char_idx < end:
                    answer_tokens.append(idx)
                    break
    
    return answer_tokens


def discover_decision_points(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    max_points: int = 3,
) -> tuple[list[DecisionPoint], torch.Tensor]:
    """
    Discover decision points via entropy/margin analysis.
    
    NO attention extraction - just logits for entropy/margin curves.
    
    Args:
        model: The language model
        input_ids: Token IDs [1, seq_len]
        tokenizer: Tokenizer for pattern matching
        max_points: Maximum decision points to return
        
    Returns:
        decision_points: List of DecisionPoint objects
        entropy: [seq_len] entropy values
    """
    device = input_ids.device
    seq_len = input_ids.shape[1]
    
    print(f"[{_ts()}] Decision point discovery (entropy/margin only, no attention)...", flush=True)
    
    # Single forward pass - NO attention output (avoids O(n²) memory)
    with torch.no_grad():
        try:
            outputs = model(input_ids, use_cache=False, output_attentions=False)
            logits = outputs.logits[0]  # [seq_len, vocab]
            del outputs
        except torch.cuda.OutOfMemoryError as e:
            print(f"[{_ts()}] FATAL: OOM even without attention output", flush=True)
            torch.cuda.empty_cache()
            raise
    
    # Compute entropy and margin curves
    entropy, margin = compute_entropy_and_margin_curves(logits)
    del logits
    
    # Find candidate decision points
    entropy_maxima = set(find_local_maxima(entropy, window=10))
    margin_minima = set(find_local_minima(margin, window=10))
    sentence_boundaries = set(find_sentence_boundaries(input_ids, tokenizer))
    answer_tokens = set(find_answer_tokens(input_ids, tokenizer))
    
    # Combine candidates
    all_candidates = entropy_maxima | margin_minima | sentence_boundaries | answer_tokens
    
    # Score and rank candidates by entropy
    candidate_scores = []
    for idx in all_candidates:
        if idx >= len(entropy) or idx < 10:  # Skip edges
            continue
        candidate_scores.append((idx, entropy[idx].item()))
    
    # Sort by entropy (higher = more uncertain = better decision point)
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Create DecisionPoint objects
    decision_points = []
    for token_idx, ent in candidate_scores[:max_points]:
        dp = DecisionPoint(
            token_idx=token_idx,
            entropy=ent,
            margin=margin[token_idx].item() if token_idx < len(margin) else 0.0,
            is_sentence_boundary=(token_idx in sentence_boundaries),
            is_answer_token=(token_idx in answer_tokens),
        )
        decision_points.append(dp)
    
    # If no candidates found, use evenly-spaced fallback
    if not decision_points:
        step = seq_len // (max_points + 1)
        for i in range(1, max_points + 1):
            idx = i * step
            decision_points.append(DecisionPoint(
                token_idx=idx,
                entropy=entropy[idx].item() if idx < len(entropy) else 0.0,
                margin=margin[idx].item() if idx < len(margin) else 0.0,
            ))
    
    print(f"[{_ts()}] Found {len(decision_points)} decision points:", flush=True)
    for dp in decision_points:
        print(f"[{_ts()}]   Token {dp.token_idx}: entropy={dp.entropy:.3f}, margin={dp.margin:.3f}", flush=True)
    
    return decision_points, entropy


def screen_chunks_by_heuristic(
    decision_points: list[DecisionPoint],
    chunk_spans: list,
    top_l: int = 15,
) -> None:
    """
    Screen chunks for each decision point using cheap heuristics.
    
    Heuristic: For each decision point, select:
    - Chunks in a recent window before the decision point
    - Plus some uniformly sampled earlier chunks
    
    Updates decision_points in-place with top_chunks field.
    
    Args:
        decision_points: List of DecisionPoint objects
        chunk_spans: List of TokenSpan objects
        top_l: Number of top chunks to keep per decision point
    """
    if not chunk_spans:
        return
    
    num_chunks = len(chunk_spans)
    
    for dp in decision_points:
        token_idx = dp.token_idx
        
        # Find chunks that end before this decision point
        prior_chunks = []
        for chunk_idx, chunk_span in enumerate(chunk_spans):
            if chunk_span.end_token <= token_idx:
                prior_chunks.append(chunk_idx)
        
        if not prior_chunks:
            # Decision point is before all chunks - use first few chunks
            dp.top_chunks = list(range(min(top_l, num_chunks)))
            continue
        
        # Heuristic: recent window (last 10) + uniform samples from earlier
        recent_window = 10
        recent_chunks = prior_chunks[-recent_window:]  # Last 10 chunks before dp
        
        # Add uniform samples from earlier chunks
        earlier_chunks = prior_chunks[:-recent_window] if len(prior_chunks) > recent_window else []
        if earlier_chunks:
            # Sample up to (top_l - len(recent)) uniformly
            n_samples = min(top_l - len(recent_chunks), len(earlier_chunks))
            if n_samples > 0:
                step = max(1, len(earlier_chunks) // n_samples)
                sampled = earlier_chunks[::step][:n_samples]
                recent_chunks = sampled + recent_chunks
        
        dp.top_chunks = recent_chunks[:top_l]
    
    total_screened = sum(len(dp.top_chunks) for dp in decision_points)
    print(f"[{_ts()}] Heuristic screening: selected {total_screened} (chunk, dp) pairs", flush=True)
