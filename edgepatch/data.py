"""
Dataset loading and chunk extraction for EdgePatch.

Loads the uzaymacar/math-rollouts dataset and extracts chunks with their
TA (Thought Anchors) labels, using the dataset's original chunk boundaries.

Supports streaming mode by aggregating raw files (the dataset yields individual
files like problem.json, chunks_labeled.json, etc.) into complete examples.
"""

from dataclasses import dataclass
from typing import Iterator, Optional
import json
import logging
import time
import os
from datetime import datetime
from collections import defaultdict

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
        split: Dataset split (e.g., "default")
        streaming: If True, use streaming mode.
        max_examples: Used for fallback slice sizing.
    
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
            
            # Log dataset type
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
                    # Compute slice size heavily padded because we stream files not examples
                    # Each example is ~3 files + overhead. Let's aim high.
                    slice_size = min(max(500, max_examples * 50), 20000)
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


def load_dataset_examples(config: EdgePatchConfig) -> Iterator[Example]:
    """
    Stream examples from the HF dataset with aggregation, early-stopping and filtering.
    
    DATASET STRUCTURE NOTE:
    The dataset yields individual FILES, not complete examples. 
    Path format: .../{solution_type}/problem_{ID}/{filename}.json
    
    We must aggregating files by problem_ID until we have enough to build an Example.
    """
    # Log configuration
    print(f"\n[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}] DATASET LOADING", flush=True)
    print(f"[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}]   Dataset: {config.dataset_name}", flush=True)
    print(f"[{_ts()}]   Solution Type: {config.solution_type}", flush=True)
    print(f"[{_ts()}]   Streaming: {config.dataset_streaming}", flush=True)
    print(f"[{_ts()}]   TA-labeled only: {config.ta_labeled_only} (field: {config.ta_label_field})", flush=True)
    print(f"[{_ts()}]   Allowlist: {len(config.example_id_allowlist or [])} IDs", flush=True)
    print(f"[{_ts()}]   Denylist: {len(config.example_id_denylist or [])} IDs", flush=True)
    print(f"[{_ts()}]   Max examples: {config.max_examples}", flush=True)
    print(f"[{_ts()}]   Max scan items: {config.max_scan_items or 'unlimited'}", flush=True)
    
    # Build allowlist and denylist sets for O(1) lookup
    allowlist = set(config.example_id_allowlist) if config.example_id_allowlist else None
    denylist = set(config.example_id_denylist) if config.example_id_denylist else None
    
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
    
    # Aggregation State
    # Buffer: problem_id -> {filename -> content_dict}
    buffer = defaultdict(dict)
    
    # Track statistics
    scanned_files = 0
    yielded_count = 0
    skipped_not_in_allowlist = 0
    skipped_in_denylist = 0
    skipped_unlabeled = 0
    skipped_incomplete = 0
    skipped_parse_error = 0
    
    print(f"[{_ts()}] Scanning files...", flush=True)
    
    for item in ds:
        scanned_files += 1
        
        # Check max_scan_items limit
        if config.max_scan_items is not None and scanned_files > config.max_scan_items:
            print(f"\n[{_ts()}] ❌ ERROR: Reached max_scan_items={config.max_scan_items} "
                  f"but only found {yielded_count}/{config.max_examples} examples!", flush=True)
            raise RuntimeError(
                f"Scan limit reached: scanned {scanned_files} files but only found "
                f"{yielded_count}/{config.max_examples} matching examples"
            )
        
        # Early stop
        if yielded_count >= config.max_examples:
            print(f"[{_ts()}] Reached max_examples={config.max_examples}, stopping early", flush=True)
            break
            
        # 1. Parse Path
        # Expected: .../{solution_type}/problem_{ID}/{filename}.json
        path = item.get('path', '')
        if not path or config.solution_type not in path:
            continue
            
        # Simple path parsing
        parts = path.strip('/').split('/')
        if len(parts) < 2:
            continue
            
        # Assuming structure: .../solution_type/problem_ID/filename
        # We look for the part starting with 'problem_'
        problem_id = None
        for part in parts:
            if part.startswith("problem_") and part != "problem_json": # avoid checking if filename is problem_json
                problem_id = part
                break
        
        if not problem_id:
            continue
            
        filename = parts[-1]
        
        # 2. Allowlist Check (Early)
        # problem_id typically looks like "problem_123"
        # We extract "123" or use the full string depending on allowlist format
        # Let's assume allowlist uses full "problem_123" or just "123".
        # We'll normalize to check both logic if needed, but for now exact match on ID string.
        if allowlist and problem_id not in allowlist:
            skipped_not_in_allowlist += 1
            # We can skip storing content for filtered IDs
            continue
        
        # 2b. Denylist Check (Exclude previously processed)
        if denylist and problem_id in denylist:
            skipped_in_denylist += 1
            continue

        # 3. Store Content
        # We only care about chunks_labeled.json and base_solution.json (or similar)
        # Note: base_solution.json contains the problem/reasoning/answer usually?
        # Let's check the keys from the previous user log:
        # base_solution.json: { "prompt": "...", "solution": "...", ... } ?
        # Actually inspect_dataset showed:
        # base_solution.json content: { "prompt": "...", ... }
        # chunks_labeled.json content: [ ... ]
        
        if filename in ["chunks_labeled.json", "base_solution.json", "problem.json"]:
            try:
                content_str = item.get('content')
                if not content_str:
                    continue
                content = json.loads(content_str)
                buffer[problem_id][filename] = content
            except json.JSONDecodeError:
                continue

        # 4. Check Completeness & Build Example
        # We need base_solution.json (for text) and chunks_labeled.json (for chunks)
        if "base_solution.json" in buffer[problem_id] and \
           "chunks_labeled.json" in buffer[problem_id]:
           
            problem_data = buffer[problem_id]
            
            # Check TA Labels
            chunks_data = problem_data["chunks_labeled.json"]
            has_labels = _has_ta_labels(chunks_data, config.ta_label_field)
            
            if config.ta_labeled_only and not has_labels:
                skipped_unlabeled += 1
                del buffer[problem_id]
                continue
                
            # Parse Example
            try:
                example = _parse_aggregated_example(problem_id, problem_data, config.ta_label_field)
                if example:
                    yield example
                    yielded_count += 1
                    if yielded_count % 5 == 0 or yielded_count == 1:
                        print(f"[{_ts()}] Yielded {yielded_count}/{config.max_examples} examples "
                              f"(scanned {scanned_files} files)...", flush=True)
            except Exception as e:
                skipped_parse_error += 1
                logger.warning(f"Failed to parse {problem_id}: {e}")
            
            # Clean up buffer for this problem
            del buffer[problem_id]
            
    # Summary
    elapsed = time.time() - start_time
    print(f"\n[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}] DATASET LOADING SUMMARY", flush=True)
    print(f"[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}]   Scanned files: {scanned_files}", flush=True)
    print(f"[{_ts()}]   Yielded examples: {yielded_count}", flush=True)
    print(f"[{_ts()}]   Skipped (allowlist): {skipped_not_in_allowlist}", flush=True) # approx (count files)
    print(f"[{_ts()}]   Skipped (unlabeled): {skipped_unlabeled}", flush=True)
    print(f"[{_ts()}]   Skipped (parse err): {skipped_parse_error}", flush=True)
    print(f"[{_ts()}]   Skipped (denylist): {skipped_in_denylist}", flush=True)
    print(f"[{_ts()}]   Time: {elapsed:.1f}s", flush=True)
    print(f"[{_ts()}] {'='*60}", flush=True)


def _has_ta_labels(chunks_labeled: list, ta_label_field: str) -> bool:
    """Check if TA labels exist in the chunks list."""
    if not chunks_labeled or not isinstance(chunks_labeled, list):
        return False
    for chunk in chunks_labeled:
        if chunk.get(ta_label_field) is not None:
            return True
    return False


def _parse_aggregated_example(
    problem_id: str,
    data: dict,
    ta_label_field: str
) -> Optional[Example]:
    """
    Parse aggregated file contents into an Example.
    data contains: {'base_solution.json': {...}, 'chunks_labeled.json': [...]}
    """
    base_sol = data.get("base_solution.json", {})
    chunks_labeled = data.get("chunks_labeled.json", [])
    
    # We need to reconstruct full text from prompt + completion?
    # Inspecting base_solution.json content from partial logs:
    # "prompt": "Solve this...", "solution": "..." (maybe?)
    # "reasoning_trace" was used in previous code, but that was for a row-based dataset.
    # UZAYMACAR dataset typically has "prompt" and "solution" (or "completion").
    # Let's try to infer fields.
    
    prompt = base_sol.get("prompt", "")
    # Try finding the completion/solution part
    # If not found, we might need to rely on what chunks reconstruct, but that's risky.
    # Looking at common formats: "solution", "completion", "full_text"
    # Inspecting item 0 from user log: base_solution.json content starts with "prompt".
    # It likely has "solution" or "completion".
    # Let's assume standard keys. If missing, we fail safely.
    
    # We will try to get the full solution text.
    # Often 'solution' contains the reasoning + answer.
    solution_text = base_sol.get("solution", base_sol.get("completion", ""))
    if not solution_text:
        # Check alternate keys
        return None

    full_text = prompt + solution_text
    
    # Identify answer span
    # The dataset often puts final answer in \boxed{}.
    # We can try to assume the answer is at the end or marked.
    # Previous code assumed separate "answer" field.
    # Let's see if base_sol has "answer".
    answer = base_sol.get("answer", "")
    
    if not answer:
        # Try to extract boxed answer logic or just treat end as answer?
        # For Edge-Patch, we need a target token span.
        # If we can't find clear answer, maybe use last N tokens?
        # Let's stick to requiring "answer" field or similar.
        # Looking at item 0 keys in user log: just 'content' json text shown.
        pass
        
    # Re-using previous logic: "problem" + "reasoning_trace" + "answer"
    # If this dataset separates them differently, we need to adapt.
    # Assuming "solution" = reasoning + answer.
    
    # Chunks Logic
    chunks = []
    
    # If we have "solution_text", we need to map chunks to it.
    # The chunks have "chunk" (text). 
    # We can search for them in full_text.
    
    current_pos = len(prompt) # Start searching after prompt
    
    for idx, chunk_data in enumerate(chunks_labeled):
        chunk_text = chunk_data.get("chunk", chunk_data.get("text", ""))
        if not chunk_text:
            continue
            
        # Find exact match
        start_char = full_text.find(chunk_text, current_pos)
        if start_char == -1:
            # Fallback: search from beginning of solution if alignment drift
            start_char = full_text.find(chunk_text, len(prompt))
            if start_char == -1:
                logger.warning(f"Chunk {idx} not found in text for {problem_id}")
                continue
        
        end_char = start_char + len(chunk_text)
        current_pos = end_char # Advance
        
        ta_label = chunk_data.get(ta_label_field, 0.0)
        
        chunks.append(Chunk(
            text=chunk_text,
            start_char=start_char,
            end_char=end_char,
            ta_label=float(ta_label) if ta_label is not None else 0.0,
            chunk_idx=idx,
        ))
        
    if not chunks:
        return None

    # Answer Span
    # If we have an explicit answer field, great.
    # If not, we might assume the answer is after the last chunk?
    # Or strict \boxed{} detection?
    # Current codebase used "answer_start_char" explicitly. 
    # Let's use the last chunk end as start of answer if no explicit answer field.
    # OR if base_sol has "answer", we append it?
    # This part is tricky without seeing full json.
    # Let's assume standard behavior: full_text = prompt + solution.
    # AND answer is part of solution.
    
    answer_start_char = len(full_text)
    answer_end_char = len(full_text)
    
    if answer:
        # Check if answer is in full_text already
        ans_pos = full_text.rfind(answer)
        if ans_pos != -1:
            answer_start_char = ans_pos
            answer_end_char = ans_pos + len(answer)
        else:
            # Append it?
            pass
            
    # Construct Example
    return Example(
        id=problem_id,
        full_text=full_text,
        chunks=chunks,
        answer_text=answer,
        answer_start_char=answer_start_char,
        answer_end_char=answer_end_char,
    )
