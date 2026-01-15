"""
Dataset loading and chunk extraction for EdgePatch.

Loads the uzaymacar/math-rollouts dataset and extracts chunks with their
TA (Thought Anchors) labels, using the dataset's original chunk boundaries.

Supports streaming mode to avoid full dataset materialization.
"""

from dataclasses import dataclass
from typing import Iterator
import json
import logging
import time
import os
from datetime import datetime

from datasets import load_dataset, IterableDataset

from edgepatch.config import EdgePatchConfig

logger = logging.getLogger("edgepatch")

# Retry configuration for HuggingFace downloads
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 60
HF_TIMEOUT_SECONDS = 120


def _ts() -> str:
    """Return current timestamp string for logging."""
    return datetime.now().strftime("%H:%M:%S")


def _load_dataset_streaming(
    dataset_name: str,
    split: str,
    streaming: bool = True,
    max_examples: int = 10,
) -> any:
    """
    Load a HuggingFace dataset with retry logic and optional streaming.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split (e.g., "train")
        streaming: If True, use streaming mode to avoid full materialization
        max_examples: Used for fallback slice sizing
    
    Returns:
        IterableDataset (streaming) or Dataset (materialized)
    """
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HF_TIMEOUT_SECONDS))
    
    last_error = None
    backoff = INITIAL_BACKOFF_SECONDS
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[{_ts()}] Loading dataset (attempt {attempt}/{MAX_RETRIES}, streaming={streaming})...", flush=True)
            ds = load_dataset(dataset_name, split=split, streaming=streaming)
            
            # Log dataset type for verification
            is_iterable = isinstance(ds, IterableDataset)
            print(f"[{_ts()}] Dataset type: {'IterableDataset (streaming)' if is_iterable else 'Dataset (materialized)'}", flush=True)
            
            return ds
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check if streaming not supported - fall back to slicing
            if streaming and ("streaming" in error_str or "iterable" in error_str):
                print(f"[{_ts()}] ⚠️ Streaming not supported, trying slice mode...", flush=True)
                try:
                    # Compute slice size: min(max(100, max_examples*200), 5000)
                    slice_size = min(max(100, max_examples * 200), 5000)
                    slice_split = f"{split}[:{slice_size}]"
                    print(f"[{_ts()}] Trying slice: {slice_split}", flush=True)
                    ds = load_dataset(dataset_name, split=slice_split)
                    print(f"[{_ts()}] Slice mode successful", flush=True)
                    return ds
                except Exception as e2:
                    print(f"[{_ts()}] ⚠️ Slice mode failed: {e2}", flush=True)
                    # Last resort: full materialize
                    print(f"[{_ts()}] ⚠️ Falling back to FULL MATERIALIZATION (this will be slow)", flush=True)
                    ds = load_dataset(dataset_name, split=split)
                    return ds
            
            # Check if this is a retryable error
            is_retryable = any(err in error_str for err in [
                "timeout", "timed out", "connection", "network",
                "temporarily unavailable", "503", "502", "429"
            ])
            
            if not is_retryable:
                print(f"[{_ts()}] Non-retryable error: {e}", flush=True)
                raise
            
            if attempt < MAX_RETRIES:
                print(
                    f"[{_ts()}] Attempt {attempt} failed: {type(e).__name__}: {str(e)[:100]}... "
                    f"Retrying in {backoff}s...",
                    flush=True
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            else:
                print(f"[{_ts()}] All {MAX_RETRIES} attempts failed", flush=True)
    
    raise RuntimeError(
        f"Failed to load dataset after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


@dataclass
class Chunk:
    """A chunk of reasoning text with its TA label."""
    text: str
    start_char: int
    end_char: int
    ta_label: float
    chunk_idx: int


@dataclass
class Example:
    """A single example with full text and chunks."""
    id: str
    full_text: str
    chunks: list[Chunk]
    answer_text: str
    answer_start_char: int
    answer_end_char: int
    
    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


def _has_ta_labels(item: dict, ta_label_field: str) -> bool:
    """
    Check if an item has TA labels based on the configured ta_label_field.
    
    The TA labels are stored per-chunk in chunks_labeled, so we check if:
    1. chunks_labeled exists and is parseable
    2. At least one chunk has a non-null value for ta_label_field
    """
    chunks_labeled_raw = item.get("chunks_labeled")
    if not chunks_labeled_raw:
        return False
    
    # Parse if string
    if isinstance(chunks_labeled_raw, str):
        try:
            chunks_labeled = json.loads(chunks_labeled_raw)
        except json.JSONDecodeError:
            return False
    else:
        chunks_labeled = chunks_labeled_raw
    
    if not chunks_labeled or not isinstance(chunks_labeled, list):
        return False
    
    # Check if at least one chunk has the ta_label_field with a non-null value
    for chunk in chunks_labeled:
        label_value = chunk.get(ta_label_field)
        if label_value is not None:
            return True
    
    return False


def load_dataset_examples(config: EdgePatchConfig) -> Iterator[Example]:
    """
    Stream examples from the HF dataset with early-stopping and filtering.
    
    Features:
    - Uses streaming mode by default (no full materialization)
    - Stops after max_examples
    - Stops after max_scan_items (if set) even if max_examples not reached
    - Filters to TA-labeled examples (if ta_labeled_only=True)
    - Filters to allowlist (if example_id_allowlist is set)
    
    Yields Example objects with:
    - full_text: concatenated problem + reasoning_trace + answer
    - chunks: list of Chunk objects from chunks_labeled
    - answer_text/answer_start_char/answer_end_char: answer span info
    """
    # Log configuration
    print(f"\n[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}] DATASET LOADING", flush=True)
    print(f"[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}]   Dataset: {config.dataset_name}", flush=True)
    print(f"[{_ts()}]   Streaming: {config.dataset_streaming}", flush=True)
    print(f"[{_ts()}]   TA-labeled only: {config.ta_labeled_only} (field: {config.ta_label_field})", flush=True)
    print(f"[{_ts()}]   Allowlist: {len(config.example_id_allowlist or [])} IDs", flush=True)
    print(f"[{_ts()}]   Max examples: {config.max_examples}", flush=True)
    print(f"[{_ts()}]   Max scan items: {config.max_scan_items or 'unlimited'}", flush=True)
    
    # Build allowlist set for O(1) lookup
    allowlist = set(config.example_id_allowlist) if config.example_id_allowlist else None
    
    # Determine streaming mode
    use_streaming = config.dataset_streaming and not config.force_materialize_dataset
    
    # Load dataset
    start_time = time.time()
    ds = _load_dataset_streaming(
        config.dataset_name,
        config.dataset_split,
        streaming=use_streaming,
        max_examples=config.max_examples,
    )
    
    # Track statistics
    scanned_count = 0
    yielded_count = 0
    skipped_not_in_allowlist = 0
    skipped_unlabeled = 0
    skipped_parse_error = 0
    
    print(f"[{_ts()}] Scanning for examples...", flush=True)
    
    for item in ds:
        scanned_count += 1
        
        # Check max_scan_items limit
        if config.max_scan_items is not None and scanned_count > config.max_scan_items:
            print(f"\n[{_ts()}] ❌ ERROR: Reached max_scan_items={config.max_scan_items} "
                  f"but only found {yielded_count}/{config.max_examples} examples!", flush=True)
            print(f"[{_ts()}] This indicates the dataset may not have enough TA-labeled examples "
                  f"in the first {config.max_scan_items} items.", flush=True)
            print(f"[{_ts()}] Solutions:", flush=True)
            print(f"[{_ts()}]   1. Increase --max-scan-items", flush=True)
            print(f"[{_ts()}]   2. Use --include-unlabeled to skip TA filtering", flush=True)
            print(f"[{_ts()}]   3. Decrease --max-examples", flush=True)
            raise RuntimeError(
                f"Scan limit reached: scanned {scanned_count} items but only found "
                f"{yielded_count}/{config.max_examples} matching examples"
            )
        
        # Early stop if we have enough examples
        if yielded_count >= config.max_examples:
            print(f"[{_ts()}] Reached max_examples={config.max_examples}, stopping early", flush=True)
            break
        
        # Allowlist filter
        example_id = item.get("id", str(hash(str(item))))
        if allowlist and example_id not in allowlist:
            skipped_not_in_allowlist += 1
            continue
        
        # TA label filter - check if chunks have the configured ta_label_field
        if config.ta_labeled_only:
            if not _has_ta_labels(item, config.ta_label_field):
                skipped_unlabeled += 1
                continue
        
        # Parse the example
        try:
            example = _parse_example(item, config.ta_label_field)
            if example is not None:
                yield example
                yielded_count += 1
                if yielded_count % 5 == 0 or yielded_count == 1:
                    print(f"[{_ts()}] Yielded {yielded_count}/{config.max_examples} examples (scanned {scanned_count})...", flush=True)
        except Exception as e:
            skipped_parse_error += 1
            logger.warning(f"Failed to parse example {example_id}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}] DATASET LOADING SUMMARY", flush=True)
    print(f"[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}]   Scanned: {scanned_count}", flush=True)
    print(f"[{_ts()}]   Yielded: {yielded_count}", flush=True)
    print(f"[{_ts()}]   Skipped (not in allowlist): {skipped_not_in_allowlist}", flush=True)
    print(f"[{_ts()}]   Skipped (unlabeled): {skipped_unlabeled}", flush=True)
    print(f"[{_ts()}]   Skipped (parse error): {skipped_parse_error}", flush=True)
    print(f"[{_ts()}]   Time: {elapsed:.1f}s", flush=True)
    print(f"[{_ts()}] {'='*60}", flush=True)


def _parse_example(item: dict, ta_label_field: str) -> Example | None:
    """
    Parse a single dataset item into an Example.
    
    Expected fields in item:
    - id: example identifier
    - problem: the problem text
    - reasoning_trace: the model's reasoning
    - answer: the final answer
    - chunks_labeled: JSON string with chunk info
    """
    example_id = item.get("id", str(hash(str(item))))
    
    # Get text components
    problem = item.get("problem", "")
    reasoning_trace = item.get("reasoning_trace", "")
    answer = item.get("answer", "")
    
    if not reasoning_trace:
        logger.debug(f"Skipping {example_id}: no reasoning_trace")
        return None
    
    # Build full text
    # Format: problem + reasoning_trace + answer
    # We need to track where each part starts
    full_text = problem + reasoning_trace + answer
    
    # Answer span in full_text
    answer_start_char = len(problem) + len(reasoning_trace)
    answer_end_char = len(full_text)
    
    # Parse chunks from chunks_labeled
    chunks_labeled_raw = item.get("chunks_labeled", None)
    if chunks_labeled_raw is None:
        logger.debug(f"Skipping {example_id}: no chunks_labeled")
        return None
    
    # Parse JSON if it's a string
    if isinstance(chunks_labeled_raw, str):
        try:
            chunks_labeled = json.loads(chunks_labeled_raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse chunks_labeled for {example_id}: {e}")
            return None
    else:
        chunks_labeled = chunks_labeled_raw
    
    # Extract chunks
    chunks = []
    reasoning_start = len(problem)  # Offset for reasoning trace in full_text
    
    for idx, chunk_data in enumerate(chunks_labeled):
        # Get chunk text and boundaries
        chunk_text = chunk_data.get("text", chunk_data.get("chunk_text", ""))
        
        # Get character offsets within reasoning_trace
        # These might be stored as start/end or start_char/end_char
        start_in_reasoning = chunk_data.get("start", chunk_data.get("start_char", None))
        end_in_reasoning = chunk_data.get("end", chunk_data.get("end_char", None))
        
        # If offsets not available, try to find the chunk in reasoning_trace
        if start_in_reasoning is None or end_in_reasoning is None:
            # Fallback: find the chunk text in reasoning_trace
            pos = reasoning_trace.find(chunk_text)
            if pos == -1:
                logger.warning(f"Could not locate chunk {idx} in reasoning_trace for {example_id}")
                continue
            start_in_reasoning = pos
            end_in_reasoning = pos + len(chunk_text)
        
        # Convert to full_text coordinates
        start_char = reasoning_start + start_in_reasoning
        end_char = reasoning_start + end_in_reasoning
        
        # Get TA label
        ta_label = chunk_data.get(ta_label_field, 0.0)
        if ta_label is None:
            ta_label = 0.0
        
        chunks.append(Chunk(
            text=chunk_text,
            start_char=start_char,
            end_char=end_char,
            ta_label=float(ta_label),
            chunk_idx=idx,
        ))
    
    if not chunks:
        logger.debug(f"Skipping {example_id}: no valid chunks")
        return None
    
    return Example(
        id=example_id,
        full_text=full_text,
        chunks=chunks,
        answer_text=answer,
        answer_start_char=answer_start_char,
        answer_end_char=answer_end_char,
    )


def get_ta_labels(example: Example) -> list[float]:
    """Extract TA labels from chunks."""
    return [chunk.ta_label for chunk in example.chunks]
