"""
Decision point discovery for rollout-light v2 scoring.

Model-driven discovery via entropy/margin curves, not patterns.
Attention-based screening for cheap chunk selection.
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
    top_chunks: list[int] = field(default_factory=list)  # Top-L chunk indices by attention
    
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
        # Fallback: approximate
        return []
    
    for match in re.finditer(r'[.!?]\s+', text):
        char_idx = match.end()
        # Find corresponding token
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
    return_attention: bool = True,
) -> tuple[list[DecisionPoint], torch.Tensor, torch.Tensor]:
    """
    Discover decision points via entropy/margin analysis.
    
    This runs ONE forward pass and extracts:
    1. Entropy/margin curves for decision point selection
    2. Attention weights for cheap chunk screening
    
    Args:
        model: The language model
        input_ids: Token IDs [1, seq_len]
        tokenizer: Tokenizer for pattern matching
        max_points: Maximum decision points to return
        return_attention: Whether to return attention weights
        
    Returns:
        decision_points: List of DecisionPoint objects
        entropy: [seq_len] entropy values
        attention_weights: [n_layers, n_heads, seq_len, seq_len] or None
    """
    device = input_ids.device
    
    # Single forward pass - try with attention, fallback without
    attention_weights = None
    
    with torch.no_grad():
        # First try to get attention weights (may fail with 4-bit or OOM)
        try:
            if return_attention:
                outputs = model(
                    input_ids,
                    use_cache=False,
                    output_attentions=True,
                )
                logits = outputs.logits[0]  # [seq_len, vocab]
                attentions = outputs.attentions
                # Stack attention weights
                attention_weights = torch.stack([a[0] for a in attentions])  # [n_layers, n_heads, seq, seq]
                print(f"[{_ts()}] Extracted attention weights: {attention_weights.shape}", flush=True)
            else:
                raise ValueError("Attention not requested")
        except Exception as e:
            print(f"[{_ts()}] Note: Could not extract attention weights ({e}), using fallback screening", flush=True)
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits[0]
    
    # Compute entropy and margin curves
    entropy, margin = compute_entropy_and_margin_curves(logits)
    
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
        seq_len = input_ids.shape[1]
        step = seq_len // (max_points + 1)
        for i in range(1, max_points + 1):
            idx = i * step
            decision_points.append(DecisionPoint(
                token_idx=idx,
                entropy=entropy[idx].item() if idx < len(entropy) else 0.0,
                margin=margin[idx].item() if idx < len(margin) else 0.0,
            ))
    
    print(f"[{_ts()}] Decision point discovery: found {len(decision_points)} points", flush=True)
    for dp in decision_points:
        print(f"[{_ts()}]   Token {dp.token_idx}: entropy={dp.entropy:.3f}, margin={dp.margin:.3f}", flush=True)
    
    return decision_points, entropy, attention_weights


def screen_chunks_by_attention(
    decision_points: list[DecisionPoint],
    attention_weights: torch.Tensor,
    chunk_spans: list,
    top_l: int = 15,
) -> None:
    """
    Screen chunks for each decision point using attention weights.
    
    This is FREE in terms of forward passes - just tensor operations.
    
    Updates decision_points in-place with top_chunks field.
    
    Args:
        decision_points: List of DecisionPoint objects
        attention_weights: [n_layers, n_heads, seq_len, seq_len] attention weights
        chunk_spans: List of TokenSpan objects
        top_l: Number of top chunks to keep per decision point
    """
    if attention_weights is None:
        # Fallback: use all chunks
        for dp in decision_points:
            dp.top_chunks = list(range(min(top_l, len(chunk_spans))))
        return
    
    # Average attention across layers and heads
    # Shape: [seq_len, seq_len]
    avg_attention = attention_weights.mean(dim=(0, 1))
    
    for dp in decision_points:
        token_idx = dp.token_idx
        
        if token_idx >= avg_attention.shape[0]:
            dp.top_chunks = list(range(min(top_l, len(chunk_spans))))
            continue
        
        # Attention from this decision token to all prior tokens
        attn_from_dp = avg_attention[token_idx, :token_idx]  # [token_idx]
        
        # Sum attention mass per chunk
        chunk_scores = []
        for chunk_idx, chunk_span in enumerate(chunk_spans):
            # Only consider chunks before the decision point
            if chunk_span.end_token > token_idx:
                continue
            
            start = chunk_span.start_token
            end = min(chunk_span.end_token, token_idx)
            
            if start < end and start < len(attn_from_dp):
                chunk_attn = attn_from_dp[start:end].sum().item()
            else:
                chunk_attn = 0.0
            
            chunk_scores.append((chunk_idx, chunk_attn))
        
        # Sort by attention and keep top-L
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        dp.top_chunks = [c[0] for c, _ in zip(chunk_scores, range(top_l))]
    
    print(f"[{_ts()}] Attention screening: top-{top_l} chunks per decision point", flush=True)
