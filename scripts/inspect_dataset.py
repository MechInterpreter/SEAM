#!/usr/bin/env python3
"""Quick script to inspect the actual HF dataset structure."""

from datasets import load_dataset

print("Loading dataset in streaming mode...")
ds = load_dataset("uzaymacar/math-rollouts", split="default", streaming=True)

print("\nFirst 3 items:")
for i, item in enumerate(ds):
    if i >= 3:
        break
    print(f"\n--- Item {i} ---")
    print(f"Keys: {list(item.keys())}")
    for key, value in item.items():
        val_str = str(value)[:200] if value else "None"
        print(f"  {key}: {val_str}...")
