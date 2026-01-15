"""
EdgePatch-Min: Minimal Edge-Patch / Causal Receiver Masking implementation.

This package provides tools for computing per-chunk causal importance via
ANSWER→CHUNK attention-edge masking with robust scoping verification.
"""

from edgepatch.config import EdgePatchConfig
from edgepatch.data import Example, Chunk, load_dataset_examples
from edgepatch.spans import TokenSpan, align_chunks_to_tokens
from edgepatch.model import load_model_and_tokenizer
from edgepatch.masking import ScopedAttentionMasker, MaskingStats
from edgepatch.scoring import compute_chunk_scores, ChunkScore
from edgepatch.eval import compute_metrics

__version__ = "0.1.0"

__all__ = [
    "EdgePatchConfig",
    "Example",
    "Chunk", 
    "load_dataset_examples",
    "TokenSpan",
    "align_chunks_to_tokens",
    "load_model_and_tokenizer",
    "ScopedAttentionMasker",
    "MaskingStats",
    "compute_chunk_scores",
    "ChunkScore",
    "compute_metrics",
]
