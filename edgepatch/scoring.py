"""
Scoring module for EdgePatch.

Computes baseline and masked log-probabilities for answer tokens,
and calculates per-chunk importance scores.
"""

from dataclasses import dataclass
from typing import Optional
import logging

import torch
import torch.nn.functional as F

from edgepatch.config import EdgePatchConfig
from edgepatch.data import Example
from edgepatch.spans import TokenSpan
from edgepatch.masking import ScopedAttentionMasker

logger = logging.getLogger("edgepatch")


@dataclass
class ChunkScore:
    """Score for a single chunk."""
    chunk_idx: int
    delta_logp: float        # masked_logP - baseline_logP
    abs_delta_logp: float    # |delta_logp|
    baseline_logp: float
    masked_logp: float
    ta_label: float          # Ground truth TA label
    
    # Instrumentation
    layers_mask_applied: Optional[list[int]] = None
    heads_mask_applied: Optional[dict] = None
    masked_entries_count: int = 0


def compute_answer_logp(
    model,
    input_ids: torch.Tensor,
    answer_start_token: int,
    answer_end_token: int,
) -> float:
    """
    Compute log-probability of answer tokens given the context.
    
    Uses teacher-forcing: the answer tokens are in the input,
    and we compute P(answer_token | all_previous_tokens) for each.
    
    Returns:
        Sum of log-probabilities for answer tokens
    """
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits  # [batch, seq_len, vocab]
    
    # For each answer token, get the log-prob from the previous position's prediction
    # P(token_i) is predicted at position i-1
    log_probs = []
    
    for i in range(answer_start_token, answer_end_token):
        if i == 0:
            continue  # Can't predict the first token
        
        # Get logits at position i-1 predicting token at position i
        logits_at_pos = logits[0, i - 1, :]  # [vocab]
        log_probs_at_pos = F.log_softmax(logits_at_pos, dim=-1)
        
        # Get the actual token at position i
        target_token = input_ids[0, i]
        
        # Get log-prob of the target token
        log_prob = log_probs_at_pos[target_token].item()
        log_probs.append(log_prob)
    
    return sum(log_probs)


def compute_chunk_scores(
    model,
    tokenizer,
    example: Example,
    chunk_spans: list[TokenSpan],
    answer_span: TokenSpan,
    model_info: dict,
    config: EdgePatchConfig,
) -> list[ChunkScore]:
    """
    Compute importance scores for each chunk.
    
    For each chunk:
    1. Run baseline forward (no masking) → baseline_logP(answer)
    2. Run masked forward (Q=answer → K=chunk blocked) → masked_logP(answer)
    3. delta = masked_logP - baseline_logP
    
    A more negative delta means the chunk was more important (blocking it hurt more).
    """
    device = next(model.parameters()).device
    
    # Tokenize the full text
    encoding = tokenizer(
        example.full_text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_len,
    )
    input_ids = encoding["input_ids"].to(device)
    
    seq_len = input_ids.shape[1]
    
    # Validate spans are within bounds
    if answer_span.end_token > seq_len:
        logger.warning(
            f"Answer span {answer_span.end_token} exceeds seq_len {seq_len}, truncating"
        )
        answer_span = TokenSpan(
            start_token=min(answer_span.start_token, seq_len - 1),
            end_token=seq_len,
            chunk_idx=answer_span.chunk_idx,
            text=answer_span.text,
        )
    
    # Compute baseline log-probability (no masking)
    baseline_logp = compute_answer_logp(
        model, input_ids, answer_span.start_token, answer_span.end_token
    )
    
    if config.verbose:
        logger.info(f"Baseline logP(answer): {baseline_logp:.4f}")
    
    # Compute masked log-probability for each chunk
    scores = []
    q_positions = list(range(answer_span.start_token, answer_span.end_token))
    
    for chunk_idx, chunk_span in enumerate(chunk_spans):
        # Skip if chunk span is out of bounds
        if chunk_span.end_token > seq_len:
            logger.warning(f"Chunk {chunk_idx} span exceeds seq_len, skipping")
            continue
        
        k_positions = list(range(chunk_span.start_token, chunk_span.end_token))
        
        # Apply scoped masking
        with ScopedAttentionMasker(
            model,
            model_info,
            config.edge_layers,
            config.edge_heads,
            validate_on_exit=True,
        ) as masker:
            masker.set_mask_positions(q_positions, k_positions)
            
            # Compute masked log-probability
            masked_logp = compute_answer_logp(
                model, input_ids, answer_span.start_token, answer_span.end_token
            )
            
            # Get instrumentation
            stats = masker.stats
        
        delta = masked_logp - baseline_logp
        
        # Probe output for chunk 0
        if chunk_idx == 0 and config.probe_chunk_0:
            logger.info(
                f"[PROBE] Chunk 0: baseline={baseline_logp:.4f}, "
                f"masked={masked_logp:.4f}, delta={delta:.4f}"
            )
            logger.info(f"[PROBE] layers_mask_applied={sorted(stats.layers_mask_applied)}")
            logger.info(f"[PROBE] heads_mask_applied={stats.heads_mask_applied}")
            logger.info(f"[PROBE] masked_entries_count={stats.masked_entries_count}")
        
        scores.append(ChunkScore(
            chunk_idx=chunk_idx,
            delta_logp=delta,
            abs_delta_logp=abs(delta),
            baseline_logp=baseline_logp,
            masked_logp=masked_logp,
            ta_label=example.chunks[chunk_idx].ta_label,
            layers_mask_applied=sorted(stats.layers_mask_applied),
            heads_mask_applied={k: sorted(v) for k, v in stats.heads_mask_applied.items()},
            masked_entries_count=stats.masked_entries_count,
        ))
        
        if config.verbose and chunk_idx % 5 == 0:
            logger.debug(
                f"Chunk {chunk_idx}/{len(chunk_spans)}: delta={delta:.4f}"
            )
    
    return scores


def get_scores_array(
    chunk_scores: list[ChunkScore],
    score_method: str = "delta_logp",
) -> list[float]:
    """Extract scores array from ChunkScore list."""
    if score_method == "delta_logp":
        # More negative = more important (blocking hurt more)
        # Negate so higher = more important for correlation
        return [-s.delta_logp for s in chunk_scores]
    elif score_method == "abs_delta_logp":
        return [s.abs_delta_logp for s in chunk_scores]
    else:
        raise ValueError(f"Unknown score_method: {score_method}")


def get_ta_labels_array(chunk_scores: list[ChunkScore]) -> list[float]:
    """Extract TA labels array from ChunkScore list."""
    return [s.ta_label for s in chunk_scores]
