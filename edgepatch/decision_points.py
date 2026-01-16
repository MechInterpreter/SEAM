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


def discover_decision_points(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    max_points: int = 3,
    chunk_spans: list = None,
) -> tuple[list[DecisionPoint], torch.Tensor]:
    """
    Discover decision points via entropy/margin analysis.
    
    NO attention extraction - just logits for entropy/margin curves.
    
    Decision points are placed AFTER the earliest chunk ends to ensure
    there are chunks that can influence them.
    
    Args:
        model: The language model
        input_ids: Token IDs [1, seq_len]
        tokenizer: Tokenizer for pattern matching
        max_points: Maximum decision points to return
        chunk_spans: Optional list of chunk spans to ensure DP placement after chunks
        
    Returns:
        decision_points: List of DecisionPoint objects
        entropy: [seq_len] entropy values
    """
    device = input_ids.device
    seq_len = input_ids.shape[1]
    
    print(f"[{_ts()}] Decision point discovery (entropy/margin only, no attention)...", flush=True)
    
    # Determine minimum position for decision points
    # Must be after at least one chunk ends to have something to screen
    min_dp_pos = 50  # Default: at least 50 tokens in
    if chunk_spans and len(chunk_spans) > 0:
        # Find where the first few chunks end
        first_chunk_ends = sorted([cs.end_token for cs in chunk_spans[:10]])
        if first_chunk_ends:
            min_dp_pos = max(min_dp_pos, first_chunk_ends[0] + 10)
    
    print(f"[{_ts()}] Min decision point position: {min_dp_pos}", flush=True)
    
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
    
    # Combine candidates, filtering by min position
    all_candidates = entropy_maxima | margin_minima
    
    # Score and rank candidates by entropy
    candidate_scores = []
    for idx in all_candidates:
        # Must be after min_dp_pos and not at the very end
        if idx < min_dp_pos or idx >= seq_len - 10:
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
        )
        decision_points.append(dp)
    
    # Fallback: if no candidates found, use evenly-spaced positions after min_dp_pos
    if not decision_points:
        print(f"[{_ts()}] No entropy peaks found, using evenly-spaced fallback", flush=True)
        available_range = seq_len - min_dp_pos - 10
        if available_range > 0:
            step = available_range // (max_points + 1)
            for i in range(1, max_points + 1):
                idx = min_dp_pos + i * step
                if idx < seq_len:
                    decision_points.append(DecisionPoint(
                        token_idx=idx,
                        entropy=entropy[idx].item() if idx < len(entropy) else 0.0,
                        margin=margin[idx].item() if idx < len(margin) else 0.0,
                    ))
    
    print(f"[{_ts()}] Found {len(decision_points)} decision points:", flush=True)
    for dp in decision_points:
        print(f"[{_ts()}]   Token {dp.token_idx}: entropy={dp.entropy:.3f}", flush=True)
    
    return decision_points, entropy


def screen_chunks_by_heuristic(
    decision_points: list[DecisionPoint],
    chunk_spans: list,
    top_l: int = 15,
) -> int:
    """
    Screen chunks for each decision point using cheap heuristics.
    
    Heuristic: For each decision point, select:
    - Chunks in a recent window before the decision point
    - Plus some uniformly sampled earlier chunks
    - Fallback: ALL chunks if nothing else works
    
    Updates decision_points in-place with top_chunks field.
    
    Args:
        decision_points: List of DecisionPoint objects
        chunk_spans: List of TokenSpan objects
        top_l: Number of top chunks to keep per decision point
        
    Returns:
        Total number of (chunk, dp) pairs selected
    """
    if not chunk_spans or not decision_points:
        print(f"[{_ts()}] WARNING: No chunks or decision points to screen", flush=True)
        return 0
    
    num_chunks = len(chunk_spans)
    total_selected = 0
    
    for dp in decision_points:
        token_idx = dp.token_idx
        
        # Find chunks that START before this decision point (can influence it)
        prior_chunks = []
        for chunk_idx, chunk_span in enumerate(chunk_spans):
            # A chunk can influence dp if it starts before dp
            # (even if it doesn't fully end before dp)
            if chunk_span.start_token < token_idx:
                prior_chunks.append(chunk_idx)
        
        print(f"[{_ts()}]   DP at token {token_idx}: {len(prior_chunks)} prior chunks", flush=True)
        
        if not prior_chunks:
            # Decision point is before all chunks - use first L chunks as fallback
            dp.top_chunks = list(range(min(top_l, num_chunks)))
            total_selected += len(dp.top_chunks)
            print(f"[{_ts()}]   -> Using first {len(dp.top_chunks)} chunks (fallback)", flush=True)
            continue
        
        # Heuristic: recent window (last 10) + uniform samples from earlier
        recent_window = 10
        if len(prior_chunks) <= top_l:
            # Use all prior chunks
            selected_chunks = prior_chunks
        else:
            # Take last `recent_window` + uniform samples from the rest
            recent_chunks = prior_chunks[-recent_window:]
            earlier_chunks = prior_chunks[:-recent_window]
            
            # Sample uniformly from earlier
            n_samples = top_l - len(recent_chunks)
            if n_samples > 0 and earlier_chunks:
                step = max(1, len(earlier_chunks) // n_samples)
                sampled = earlier_chunks[::step][:n_samples]
                selected_chunks = sampled + recent_chunks
            else:
                selected_chunks = recent_chunks
        
        dp.top_chunks = selected_chunks[:top_l]
        total_selected += len(dp.top_chunks)
        print(f"[{_ts()}]   -> Selected {len(dp.top_chunks)} chunks", flush=True)
    
    print(f"[{_ts()}] Heuristic screening: {total_selected} total (chunk, dp) pairs", flush=True)
    return total_selected
