# EdgePatch / SEAM Walkthrough

This document explains exactly how the Edge-Patch implementation works, with a focus on ensuring scoped masking is correct.

## How Input Text Is Constructed

The input text is constructed from three components in the HF dataset:

```
full_text = problem + reasoning_trace + answer
```

- **problem**: The math problem statement
- **reasoning_trace**: The model's step-by-step reasoning
- **answer**: The final answer

The character offsets are tracked:
- `answer_start_char = len(problem) + len(reasoning_trace)`
- `answer_end_char = len(full_text)`

## How Chunk Spans Are Aligned

### Source of Chunks

Chunks come from the `chunks_labeled` field in the HF dataset. This is a JSON string containing:
```json
[
  {"text": "chunk text", "start": 0, "end": 50, "counterfactual_importance_accuracy": 0.8},
  ...
]
```

**Important**: We use the dataset's original chunk boundaries. NO re-splitting.

### Alignment Process

The alignment happens in `edgepatch/spans.py`:

1. **Primary Method (offset mapping)**:
   ```python
   encoding = tokenizer(text, return_offsets_mapping=True)
   # offset_mapping gives (start_char, end_char) for each token
   # We find tokens that overlap with chunk's character span
   ```

2. **Fallback Method (incremental tokenization)**:
   ```python
   start_tok = len(tokenize(prefix_text))
   end_tok = len(tokenize(prefix_text + chunk_text))
   ```

3. **Hard-fail on ambiguity**: If alignment fails, we raise `AlignmentError`. No heuristic fallbacks.

## Where Exactly Masking Is Applied

### Location in Code

The masking happens in `edgepatch/masking.py`, specifically in `ScopedAttentionMasker._create_patched_forward()`.

### The Masking Process

1. **Pre-scan**: At model load time, we find each layer's attention module:
   ```python
   for layer_idx, layer in enumerate(model.model.layers):
       attn_module = layer.self_attn
       attn_module._edgepatch_layer_idx = layer_idx
   ```

2. **Patch attention forward**: We replace each attention module's `forward()` method:
   ```python
   attn_module.forward = self._create_patched_forward(original_forward, layer_idx)
   ```

3. **Inside patched forward**: For selected layers only:
   ```python
   # Create edge mask
   edge_mask[batch, heads, q_positions, k_positions] = -inf
   
   # Combine with existing attention mask
   combined_mask = attention_mask + edge_mask
   
   # Call original forward with modified mask
   return original_forward(..., attention_mask=combined_mask, ...)
   ```

### What Gets Masked

- **Q positions**: Token indices of the answer
- **K positions**: Token indices of the chunk being tested
- **Effect**: Answer tokens cannot attend to chunk tokens at specified layers/heads

## How Scoping Is Verified

### Runtime Instrumentation

During each masked forward pass, we record:

```python
@dataclass
class MaskingStats:
    layers_hit: set           # All layers whose forward was called
    layers_mask_applied: set  # Layers where mask was actually applied
    heads_mask_applied: dict  # layer_idx -> set of head indices
    masked_entries_count: int # Total attention entries set to -inf
```

### Hard-Fail Validation

After each forward pass, `_validate_instrumentation()` checks:

```python
if self.edge_layers is not None:
    expected_layers = self.edge_layers & self.stats.layers_hit
    if self.stats.layers_mask_applied != expected_layers:
        raise MaskingError("MASKING VALIDATION FAILED!")
```

### Toggle Tests

The Colab notebook includes explicit tests:

**Layer Toggle Test**:
```python
scores_A = run_with_layers([0], ...)  # Mask layer 0
scores_B = run_with_layers([24,25,26,27,28,29,30,31], ...)  # Mask layers 24-31

max_diff = max(abs(scores_A - scores_B))
assert max_diff > 1e-6, "LAYER TOGGLE FAILED!"
```

**Head Toggle Test**:
```python
scores_A = run_with_heads([0], ...)  # Mask head 0
scores_B = run_with_heads([1], ...)  # Mask head 1

max_diff = max(abs(scores_A - scores_B))
assert max_diff > 1e-6, "HEAD TOGGLE FAILED!"
```

## Exact Colab Commands

### Setup
```bash
!git clone https://github.com/MechInterpreter/SEAM.git
%cd SEAM
!pip install -e .
```

### Smoke Test (1 example)
```bash
!python scripts/run_edgepatch.py smoke
```

### Layer Toggle Test
```bash
!python scripts/run_edgepatch.py smoke --edge-layers 0 --output-dir runs/layer_test_A
!python scripts/run_edgepatch.py smoke --edge-layers 24 25 26 27 28 29 30 31 --output-dir runs/layer_test_B
```

### Head Toggle Test
```bash
!python scripts/run_edgepatch.py smoke --edge-layers 0 1 2 3 --edge-heads 0 --output-dir runs/head_test_A
!python scripts/run_edgepatch.py smoke --edge-layers 0 1 2 3 --edge-heads 1 --output-dir runs/head_test_B
```

### Confirm Run (3 examples)
```bash
!python scripts/run_edgepatch.py confirm
```

## Artifacts

Each run saves to `runs/edgepatch_YYYYMMDD_HHMMSS/`:

- `config.json` - Run configuration
- `run_summary.json` - Run metadata and status
- `all_results.json` - Per-example, per-chunk scores
- `eval_metrics.json` - Aggregate evaluation metrics

## Known Limitations

1. **Eager attention required**: Flash Attention / SDPA not supported (masking requires pre-softmax access)
2. **Memory**: Long contexts may require gradient checkpointing
3. **Speed**: Masking adds overhead; not optimized for speed

## Troubleshooting

### "MaskingError: MASKING VALIDATION FAILED!"
The masking layer/head specification didn't match what was actually masked. Check:
- Are the layer indices valid for the model?
- Is eager attention enabled?

### "AlignmentError"
Chunk-to-token alignment failed. Check:
- Is the chunk text present in the full text?
- Are character offsets correct in the dataset?

### Identical scores for different layers/heads
This is the failure mode we designed against. If you see this:
1. Check that `attn_implementation="eager"` is set
2. Check that the patched forward is actually being called (add debug prints)
3. Run the toggle tests to isolate the issue

## Robustness Update (Jan 15)
We have hardened the pipeline against common failure modes:

1. **Alignment Failure Handling**:
   - The runner now automatically **resamples** examples when it encounters chunk alignment failures (e.g., character offsets not matching token boundaries).
   - It will keep trying until the requested `max_examples` are successfully scored, capping at 50 consecutive failures to prevent infinite loops.
   - Failed examples are logged to `failed_examples.jsonl` for offline debugging.

2. **Scientific Controls**:
   - Validation notebook cells (Smoke, Layer Toggle, Head Toggle) now use a **Pinned Example** (`problem_1591`) and **Extended Scoring Span** to ensuring strictly comparable results across runs.
   - Large-scale runs (`confirm`, `main` modes) automatically use random streaming for coverage.

3. **Bug Fix (Masked LogP = 0)**:
   - Fixed a critical bug where `scoring_start` (token index) was shadowed by a timing variable, causing the masked pass to iterate over an empty range.
