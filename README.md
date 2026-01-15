# EdgePatch / SEAM: Scoped Edge Attention Masking

Minimal implementation of Edge-Patch / Causal Receiver Masking for computing per-chunk importance in reasoning traces.

## Quick Start

### Local Installation
```bash
git clone https://github.com/MechInterpreter/SEAM.git
cd SEAM
pip install -e .
```

### Colab One-Liner
```python
!git clone https://github.com/MechInterpreter/SEAM.git && cd SEAM && pip install -e .
```

### Run Smoke Test
```bash
python scripts/run_edgepatch.py smoke
```

### Run with Layer/Head Specification
```bash
# Mask only layers 0-3
python scripts/run_edgepatch.py smoke --edge-layers 0 1 2 3

# Mask only heads 0 and 1 in all layers
python scripts/run_edgepatch.py smoke --edge-heads 0 1
```

## Features

- **Dataset**: Loads `uzaymacar/math-rollouts` with pre-computed chunk boundaries
- **Model**: DeepSeek-R1-Distill-Llama-8B with 4-bit quantization
- **Masking**: Per-layer, per-head scoped attention masking
- **Evaluation**: Spearman correlation, top-k overlap, PR-AUC vs TA labels

## Critical Design: Scoped Masking with Verification

This implementation is designed to make it **impossible** for `edge_layers`/`edge_heads` to be ignored:

1. **Runtime Instrumentation**: Every masked forward records `layers_hit`, `layers_mask_applied`, `heads_mask_applied`
2. **Hard-Fail Validation**: Post-forward validation raises `MaskingError` if actual != expected
3. **Toggle Tests**: The notebook includes explicit tests that different layer/head configs produce different scores

## Repository Structure

```
SEAM/
├── edgepatch/
│   ├── __init__.py
│   ├── config.py      # Configuration dataclass
│   ├── data.py        # Dataset loading
│   ├── spans.py       # Chunk→token alignment
│   ├── model.py       # Model loading
│   ├── masking.py     # CRITICAL: Scoped masking with instrumentation
│   ├── scoring.py     # Log-probability computation
│   ├── eval.py        # Metrics
│   └── utils.py       # Utilities
├── scripts/
│   └── run_edgepatch.py
├── notebooks/
│   └── EdgePatch_Colab.ipynb
├── tests/
│   ├── test_span_alignment.py
│   └── test_scoping_sanity.py
├── walkthrough.md
├── pyproject.toml
└── requirements.txt
```

## Requirements

- Python ≥3.10
- transformers==4.57.3 (pinned)
- torch ≥2.0.0
- CUDA-capable GPU (tested on A100)

## License

MIT
