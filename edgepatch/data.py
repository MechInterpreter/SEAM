"""
Dataset loading and chunk extraction for EdgePatch.

Loads the uzaymacar/math-rollouts dataset and extracts chunks with their
TA (Thought Anchors) labels, using the dataset's original chunk boundaries.
"""

from dataclasses import dataclass
from typing import Iterator
import json
import logging

from datasets import load_dataset

from edgepatch.config import EdgePatchConfig

logger = logging.getLogger("edgepatch")


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
    Load examples from the HF dataset.
    
    Yields Example objects with:
    - full_text: concatenated problem + reasoning_trace + answer
    - chunks: list of Chunk objects from chunks_labeled.json
    - answer_text/answer_start_char/answer_end_char: answer span info
    
    Uses the dataset's original chunk boundaries (NO re-splitting).
    """
    logger.info(f"Loading dataset: {config.dataset_name} (split: {config.dataset_split})")
    
    ds = load_dataset(config.dataset_name, split=config.dataset_split)
    
    count = 0
    for item in ds:
        if count >= config.max_examples:
            break
        
        try:
            example = _parse_example(item, config.ta_label_field)
            if example is not None:
                yield example
                count += 1
        except Exception as e:
            logger.warning(f"Failed to parse example {item.get('id', 'unknown')}: {e}")
            continue
    
    logger.info(f"Loaded {count} examples")


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
    
    # Parse chunks from chunks_labeled.json
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
