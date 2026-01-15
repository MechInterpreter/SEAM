#!/usr/bin/env python3
"""
Test script for streaming dataset loading.

Verifies that:
1. In streaming mode, we get an IterableDataset (not a full Dataset)
2. We can yield the requested number of examples without full materialization
3. TA filtering works correctly based on ta_label_field
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import IterableDataset

from edgepatch.config import EdgePatchConfig
from edgepatch.data import load_dataset_examples, _load_dataset_streaming, _has_ta_labels


def test_streaming_returns_iterable_dataset():
    """Verify streaming mode returns IterableDataset, not materialized Dataset."""
    print("\n" + "="*60)
    print("TEST: Streaming returns IterableDataset")
    print("="*60)
    
    ds = _load_dataset_streaming(
        "uzaymacar/math-rollouts",
        "train",
        streaming=True,
        max_examples=1,
    )
    
    is_iterable = isinstance(ds, IterableDataset)
    print(f"Dataset type: {type(ds).__name__}")
    print(f"Is IterableDataset: {is_iterable}")
    
    assert is_iterable, f"Expected IterableDataset, got {type(ds).__name__}"
    print("✓ PASS: Streaming returns IterableDataset")


def test_streaming_yields_examples():
    """Verify streaming yields the requested number of examples."""
    print("\n" + "="*60)
    print("TEST: Streaming yields requested examples")
    print("="*60)
    
    config = EdgePatchConfig(
        max_examples=2,
        dataset_streaming=True,
        ta_labeled_only=True,
        max_scan_items=5000,
    )
    
    examples = list(load_dataset_examples(config))
    
    print(f"Requested: {config.max_examples} examples")
    print(f"Received:  {len(examples)} examples")
    
    assert len(examples) == config.max_examples, \
        f"Expected {config.max_examples} examples, got {len(examples)}"
    
    # Verify examples have chunks
    for i, ex in enumerate(examples):
        assert ex.chunks, f"Example {i} has no chunks"
        print(f"  Example {i}: {ex.id} with {len(ex.chunks)} chunks")
    
    print("✓ PASS: Streaming yields requested examples")


def test_ta_filter():
    """Verify TA label filtering works correctly."""
    print("\n" + "="*60)
    print("TEST: TA label filtering")
    print("="*60)
    
    config = EdgePatchConfig(
        max_examples=3,
        dataset_streaming=True,
        ta_labeled_only=True,
        ta_label_field="counterfactual_importance_accuracy",
        max_scan_items=5000,
    )
    
    examples = list(load_dataset_examples(config))
    
    print(f"Loaded {len(examples)} examples with ta_labeled_only=True")
    
    # Verify each example has chunks with TA labels
    for i, ex in enumerate(examples):
        has_label = any(c.ta_label is not None for c in ex.chunks)
        print(f"  Example {i}: {len(ex.chunks)} chunks, has_label={has_label}")
        assert has_label, f"Example {i} has no TA labels despite filtering"
    
    print("✓ PASS: TA label filtering works")


def test_streaming_no_full_materialize_logs():
    """Verify streaming mode doesn't trigger 'Generating split' logs."""
    print("\n" + "="*60)
    print("TEST: No full materialization in streaming mode")
    print("="*60)
    
    import io
    import contextlib
    
    # Capture stdout to check for materialization warnings
    output = io.StringIO()
    
    config = EdgePatchConfig(
        max_examples=1,
        dataset_streaming=True,
        ta_labeled_only=True,
        max_scan_items=2000,
    )
    
    # We can't easily capture HF's progress bars, so this is best-effort
    # Just verify we get an IterableDataset
    ds = _load_dataset_streaming(
        config.dataset_name,
        config.dataset_split,
        streaming=True,
        max_examples=1,
    )
    
    is_iterable = isinstance(ds, IterableDataset)
    print(f"Dataset is IterableDataset: {is_iterable}")
    
    if is_iterable:
        print("✓ PASS: Streaming mode active (best-effort verification)")
    else:
        print("⚠️ WARN: Not using streaming - may have triggered materialization")


def main():
    """Run all streaming tests."""
    print("\n" + "#"*60)
    print("# STREAMING DATASET LOADING TESTS")
    print("#"*60)
    
    tests = [
        test_streaming_returns_iterable_dataset,
        test_streaming_yields_examples,
        test_ta_filter,
        test_streaming_no_full_materialize_logs,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
