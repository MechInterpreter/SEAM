"""
Scoring module for EdgePatch.

Computes baseline and masked log-probabilities for answer tokens,
and calculates per-chunk importance scores.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import time
import threading
from datetime import datetime

import torch
import torch.nn.functional as F

from edgepatch.config import EdgePatchConfig
from edgepatch.data import Example
from edgepatch.spans import TokenSpan
from edgepatch.masking import ScopedAttentionMasker

logger = logging.getLogger("edgepatch")


def _ts() -> str:
    """Return current timestamp string for logging."""
    return datetime.now().strftime("%H:%M:%S")


class Heartbeat:
    """
    Background heartbeat thread that prints a message every N seconds.
    
    Confirms the process is still alive even if no chunk has finished.
    """
    
    def __init__(self, interval_seconds: float = 30.0, context: str = ""):
        self.interval = interval_seconds
        self.context = context
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._current_chunk = 0
        self._total_chunks = 0
    
    def update_progress(self, current_chunk: int, total_chunks: int):
        """Update current progress for heartbeat messages."""
        self._current_chunk = current_chunk
        self._total_chunks = total_chunks
    
    def _run(self):
        while not self._stop_event.wait(self.interval):
            elapsed = time.time() - self._start_time
            progress = f"{self._current_chunk}/{self._total_chunks}" if self._total_chunks > 0 else "starting"
            print(f"[{_ts()}] 💓 HEARTBEAT: {self.context} | progress={progress} | elapsed={elapsed:.1f}s", flush=True)
    
    def start(self):
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)


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
    tokenizer=None,
    device=None,
    return_probs: bool = False
):
    """
    Compute log-probability of answer tokens given the context.
    
    Args:
        return_probs: If True, also return list of per-token probabilities for saturation check.
    
    Returns:
        If return_probs=False: float (sum of log-probs)
        If return_probs=True: tuple (float sum_logp, list[float] probs)
    """
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits  # [batch, seq_len, vocab]
    
    # For each answer token, get the log-prob from the previous position's prediction
    log_probs = []
    per_token_probs = []
    
    # Only print diagnostics if tokenizer provided (usually for baseline)
    show_diagnostics = (tokenizer is not None)
    
    if show_diagnostics:
        print(f"\n[{_ts()}] [DIAGNOSTIC] Scoring Span (tokens {answer_start_token}-{answer_end_token}):", flush=True)
    
    for i in range(answer_start_token, answer_end_token):
        if i == 0:
            continue
        
        # Get logits at position i-1 predicting token at position i
        logits_at_pos = logits[0, i - 1, :].to(dtype=torch.float32)
        log_probs_at_pos = F.log_softmax(logits_at_pos, dim=-1)
        
        # Target
        target_token_id = input_ids[0, i].item()
        log_prob = log_probs_at_pos[target_token_id].item()
        log_probs.append(log_prob)
        
        # Compute probability for saturation check
        probs = F.softmax(logits_at_pos, dim=-1)
        target_prob = probs[target_token_id].item()
        per_token_probs.append(target_prob)
        
        # Detailed diagnostics
        if show_diagnostics:
            target_str = tokenizer.decode([target_token_id])
            
            # Rank
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            rank = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0].item() + 1
            
            # Top-3 info
            top3_indices = sorted_indices[:3]
            top3_info = []
            for idx in top3_indices:
                tok_str = tokenizer.decode([idx]).replace('\n', '\\n')
                prob_val = probs[idx].item()
                top3_info.append(f"'{tok_str}' ({prob_val:.4f})")
            
            print(f"  Pos {i} | Target: {repr(target_str)} (id: {target_token_id}) | "
                  f"LogP: {log_prob:.4f} | Prob: {target_prob:.4f} | Rank: {rank}", flush=True)
            print(f"    Top-3: {', '.join(top3_info)}", flush=True)
    
    if return_probs:
        return sum(log_probs), per_token_probs
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
    
    # ================================================================
    # STARTUP BANNER
    # ================================================================
    num_chunks = len(chunk_spans)
    num_layers = model_info.get("num_layers", 32)
    num_heads = model_info.get("num_heads", 32)
    
    # Determine actual layers/heads being tested
    layers_to_test = config.edge_layers if config.edge_layers else list(range(num_layers))
    heads_to_test = config.edge_heads if config.edge_heads else list(range(num_heads))
    
    # Estimate forward passes: 1 baseline + N chunks
    # (Each chunk = 1 forward pass through model with patched attention)
    estimated_forwards = 1 + num_chunks
    
    print(f"\n[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}] SCORING CONFIGURATION", flush=True)
    print(f"[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}]   Chunks to score:        {num_chunks}", flush=True)
    print(f"[{_ts()}]   Layers being masked:    {len(layers_to_test)} (indices: {layers_to_test[:5]}{'...' if len(layers_to_test) > 5 else ''})", flush=True)
    print(f"[{_ts()}]   Heads being masked:     {len(heads_to_test)} (indices: {heads_to_test[:5]}{'...' if len(heads_to_test) > 5 else ''})", flush=True)
    print(f"[{_ts()}]   Estimated forward passes: {estimated_forwards}", flush=True)
    print(f"[{_ts()}] {'='*60}\n", flush=True)
    
    # ================================================================
    # TOKENIZATION PHASE
    # ================================================================
    print(f"[{_ts()}] PHASE: Tokenization...", flush=True)
    tok_start = time.time()
    
    encoding = tokenizer(
        example.full_text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_seq_len,
    )
    input_ids = encoding["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    
    tok_elapsed = time.time() - tok_start
    print(f"[{_ts()}] PHASE: Tokenization complete ({tok_elapsed:.2f}s) | seq_len={seq_len}", flush=True)
    
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
    
    # ================================================================
    # SCORING SPAN CALCULATION
    # ================================================================
    # Determine actual span to score based on config
    scoring_start = answer_span.start_token
    scoring_end = answer_span.end_token
    
    if config.score_span == "extended":
        # Extend backwards to include more tokens (less saturated)
        scoring_start = max(1, answer_span.start_token - config.score_extend_tokens)
        print(f"[{_ts()}] Extended scoring span: tokens {scoring_start}-{scoring_end} "
              f"(+{answer_span.start_token - scoring_start} reasoning tokens)", flush=True)
    elif config.score_span == "reasoning_only":
        # Score only the tokens before the answer (last chunk of reasoning)
        scoring_end = answer_span.start_token
        scoring_start = max(1, scoring_end - config.score_extend_tokens)
        print(f"[{_ts()}] Reasoning-only span: tokens {scoring_start}-{scoring_end}", flush=True)
    else:  # answer_only
        print(f"[{_ts()}] Answer-only span: tokens {scoring_start}-{scoring_end}", flush=True)
    
    # ================================================================
    # BASELINE FORWARD PASS
    # ================================================================
    print(f"[{_ts()}] PHASE: Computing baseline logP (1 forward pass)...", flush=True)
    baseline_start = time.time()
    
    baseline_logp, baseline_probs = compute_answer_logp(
        model, input_ids, scoring_start, scoring_end,
        tokenizer=tokenizer,  # Enable diagnostics for baseline
        return_probs=True     # Also return per-token probs for saturation check
    )
    
    baseline_elapsed = time.time() - baseline_start
    print(f"[{_ts()}] PHASE: Baseline complete ({baseline_elapsed:.2f}s) | logP={baseline_logp:.4f}", flush=True)
    
    # Saturation check
    if baseline_probs and all(p > config.saturation_threshold for p in baseline_probs):
        print(f"\n[{_ts()}] ⚠️ SATURATION WARNING: All baseline probs > {config.saturation_threshold}", flush=True)
        print(f"[{_ts()}] ⚠️ Deltas will be uninformative. Consider using --score-span extended", flush=True)
        print(f"[{_ts()}] ⚠️ Probs: {[f'{p:.4f}' for p in baseline_probs]}", flush=True)
    else:
        min_prob = min(baseline_probs) if baseline_probs else 0
        print(f"[{_ts()}] ✓ Non-saturated target (min prob: {min_prob:.4f})", flush=True)
    
    # ================================================================
    # MASKED FORWARD PASSES (main loop)
    # ================================================================
    print(f"\n[{_ts()}] PHASE: Scoring {num_chunks} chunks (masked forward passes)...", flush=True)
    
    scores = []
    q_positions = list(range(scoring_start, scoring_end))  # Use extended span
    
    # Start heartbeat
    heartbeat = Heartbeat(interval_seconds=30.0, context=f"Scoring example {example.id}")
    heartbeat.update_progress(0, num_chunks)
    heartbeat.start()
    
    scoring_start = time.time()
    
    try:
        for chunk_idx, chunk_span in enumerate(chunk_spans):
            chunk_start = time.time()
            
            # Update heartbeat
            heartbeat.update_progress(chunk_idx, num_chunks)
            
            # Progress log for every chunk
            print(f"[{_ts()}] Processing chunk {chunk_idx + 1}/{num_chunks} (tokens {chunk_span.start_token}-{chunk_span.end_token})...", flush=True)
            
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
            chunk_elapsed = time.time() - chunk_start
            
            # Detailed log for first chunk
            if chunk_idx == 0 and config.probe_chunk_0:
                print(f"[{_ts()}] [PROBE] Chunk 0: baseline={baseline_logp:.4f}, masked={masked_logp:.4f}, delta={delta:.4f}", flush=True)
                print(f"[{_ts()}] [PROBE] layers_mask_applied={sorted(stats.layers_mask_applied)}", flush=True)
                print(f"[{_ts()}] [PROBE] heads_mask_applied={stats.heads_mask_applied}", flush=True)
                print(f"[{_ts()}] [PROBE] masked_entries_count={stats.masked_entries_count}", flush=True)
            
            # Progress summary every chunk
            print(f"[{_ts()}] Chunk {chunk_idx + 1}/{num_chunks} complete ({chunk_elapsed:.2f}s) | delta={delta:.4f}", flush=True)
            
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
    
    finally:
        heartbeat.stop()
    
    scoring_elapsed = time.time() - scoring_start
    
    print(f"\n[{_ts()}] PHASE: Scoring complete", flush=True)
    print(f"[{_ts()}]   Total chunks scored: {len(scores)}", flush=True)
    print(f"[{_ts()}]   Total scoring time:  {scoring_elapsed:.2f}s", flush=True)
    print(f"[{_ts()}]   Avg time per chunk:  {scoring_elapsed/max(len(scores),1):.2f}s", flush=True)
    
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
