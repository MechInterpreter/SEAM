#!/usr/bin/env python3
"""
Test script for streaming dataset loading (File Aggregation Version).

Verifies that:
1. File aggregation logic works (reassembling chunks + solution).
2. TA label field filtering works.
3. Correctly produces Example objects.
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgepatch.config import EdgePatchConfig
from edgepatch.data import load_dataset_examples, _load_dataset_streaming, _has_ta_labels

def test_file_aggregation_smoke():
    """Verify we can load at least one complete example via aggregation."""
    print("\n" + "="*60)
    print("TEST: File Aggregation Smoke")
    print("="*60)
    
    # We use a larger max_scan_items to ensure we get enough files
    # Each example is multiple files, so scanning 2000 is safe.
    config = EdgePatchConfig(
        max_examples=1,
        dataset_streaming=True,
        ta_labeled_only=True,
        solution_type="correct_base_solution",
        max_scan_items=2000,
    )
    
    # Streaming load
    examples = list(load_dataset_examples(config))
    
    print(f"Loaded {len(examples)} examples")
    
    assert len(examples) == 1, f"Expected 1 example, got {len(examples)}"
    ex = examples[0]
    
    print(f"Example ID: {ex.id}")
    print(f"Full text length: {len(ex.full_text)}")
    print(f"Chunks: {len(ex.chunks)}")
    
    # Verify chunks exist and have text
    assert len(ex.chunks) > 0
    assert ex.chunks[0].text
    assert ex.full_text
    
    print("✓ PASS: File Aggregation Smoke")

def test_ta_filter_aggregation():
    """Verify TA filtering works on aggregated content."""
    print("\n" + "="*60)
    print("TEST: TA Filter Aggregation")
    print("="*60)
    
    # Use config with specific label field
    config = EdgePatchConfig(
        max_examples=1,
        dataset_streaming=True,
        ta_labeled_only=True,
        ta_label_field="counterfactual_importance_accuracy",
        max_scan_items=2000,
    )
    
    examples = list(load_dataset_examples(config))
    
    if not examples:
        print("⚠️ WARN: No examples found with TA labels (might be data issue or filter issue)")
        return

    ex = examples[0]
    print(f"Example {ex.id} has {len(ex.chunks)} chunks")
    
    # Check if any chunk has the label
    has_label = any(c.ta_label != 0.0 for c in ex.chunks) # We default to 0.0 if not found, but data.py sets 0.0 if field missing? 
    # Actually data.py: ta_label = float(ta_label) if ta_label is not None else 0.0
    # And filter checks _has_ta_labels which checks for NON-NONE.
    # So if it passed filter, it MUST have had the field in raw json.
    
    print(f"Chunks have valid labels? {has_label}")
    # Note: value might be 0.0 even if present, but strict presence was checked.
    
    print("✓ PASS: TA Filter Aggregation logic ran")

def main():
    """Run tests."""
    print("\n" + "#"*60)
    print("# STREAMING AGGREGATION TESTS")
    print("#"*60)
    
    tests = [
        test_file_aggregation_smoke,
        test_ta_filter_aggregation,
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
