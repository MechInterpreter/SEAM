#!/usr/bin/env python3
"""
EdgePatch main runner script.

Usage:
    python scripts/run_edgepatch.py smoke                    # Quick test (1 example)
    python scripts/run_edgepatch.py confirm                  # Short run (3 examples)
    python scripts/run_edgepatch.py main --max-examples 10   # Full run

    # With layer/head specification
    python scripts/run_edgepatch.py smoke --edge-layers 0 1 2 3
    python scripts/run_edgepatch.py smoke --edge-heads 0
"""

import sys
import os
import time
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgepatch.config import EdgePatchConfig, create_arg_parser
from edgepatch.data import load_dataset_examples, Example
from edgepatch.spans import align_chunks_to_tokens, validate_spans
from edgepatch.model import load_model_and_tokenizer
from edgepatch.scoring import compute_chunk_scores, ChunkScore
from edgepatch.eval import compute_metrics, print_metrics_summary
from edgepatch.utils import setup_logging, get_run_dir, RunArtifacts


def _ts() -> str:
    """Return current timestamp string for logging."""
    return datetime.now().strftime("%H:%M:%S")


def main():
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Load config
    if args.config:
        base_config = EdgePatchConfig.from_yaml(args.config)
    else:
        base_config = EdgePatchConfig()
    
    # Apply mode-specific defaults
    if args.mode == "smoke":
        if args.max_examples is None:
            args.max_examples = 1
        # Set default max_scan_items for smoke
        if not hasattr(args, 'max_scan_items') or getattr(args, 'max_scan_items', None) is None:
            base_config.max_scan_items = 2000
            
        # Set default layers/heads for smoke testing (diagnostic)
        # Use mid-layers to ensure we see non-zero deltas if masking works
        if args.edge_layers is None:
            args.edge_layers = [15, 16]
        if args.edge_heads is None:
            args.edge_heads = [0]
    elif args.mode == "confirm":
        if args.max_examples is None:
            args.max_examples = 3
        # Set default max_scan_items for confirm
        if not hasattr(args, 'max_scan_items') or getattr(args, 'max_scan_items', None) is None:
            base_config.max_scan_items = 5000
    # main mode: max_scan_items stays None (unlimited)
    
    # Override with CLI args
    config = EdgePatchConfig.from_args(args, base_config)
    
    # Setup logging
    logger = setup_logging(config.verbose)
    
    # Create run directory
    run_dir = get_run_dir(config.output_dir, config.run_name)
    artifacts = RunArtifacts(run_dir)
    
    print(f"\n[{_ts()}] {'='*60}", flush=True)
    print(f"[{_ts()}] EdgePatch Run: {args.mode.upper()}", flush=True)
    print(f"[{_ts()}] Output: {run_dir}", flush=True)
    print(f"[{_ts()}] {'='*60}", flush=True)
    
    # Streaming/filtering settings
    print(f"[{_ts()}] Dataset Settings:", flush=True)
    print(f"[{_ts()}]   Streaming: {config.dataset_streaming}", flush=True)
    print(f"[{_ts()}]   TA-labeled only: {config.ta_labeled_only} (field: {config.ta_label_field})", flush=True)
    print(f"[{_ts()}]   Allowlist: {len(config.example_id_allowlist or [])} IDs", flush=True)
    print(f"[{_ts()}]   Max examples: {config.max_examples}", flush=True)
    print(f"[{_ts()}]   Max scan items: {config.max_scan_items or 'unlimited'}", flush=True)
    
    # Edge masking config (CRITICAL)
    print(f"[{_ts()}] Edge Masking:", flush=True)
    print(f"[{_ts()}]   Layers: {config.edge_layers or 'ALL'}", flush=True)
    print(f"[{_ts()}]   Heads: {config.edge_heads or 'ALL'}", flush=True)
    
    # Save config
    artifacts.save_config(config.to_dict())
    
    start_time = time.time()
    
    try:
        # ================================================================
        # PHASE 1: MODEL LOADING
        # ================================================================
        print(f"\n[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] PHASE 1: MODEL LOADING", flush=True)
        print(f"[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] Loading {config.model_name}...", flush=True)
        print(f"[{_ts()}] (This may take 2-3 minutes on first run)", flush=True)
        
        model_start = time.time()
        model, tokenizer, model_info = load_model_and_tokenizer(config)
        model_elapsed = time.time() - model_start
        
        print(f"[{_ts()}] Model loaded in {model_elapsed:.1f}s", flush=True)
        print(f"[{_ts()}]   Layers: {model_info['num_layers']}", flush=True)
        print(f"[{_ts()}]   Heads:  {model_info['num_heads']}", flush=True)
        
        # ================================================================
        # PHASE 2: DATASET LOADING
        # ================================================================
        print(f"\n[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] PHASE 2: DATASET LOADING", flush=True)
        print(f"[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] Loading dataset: {config.dataset_name}...", flush=True)
        
        dataset_start = time.time()
        examples = list(load_dataset_examples(config))
        dataset_elapsed = time.time() - dataset_start
        
        print(f"[{_ts()}] Dataset loaded in {dataset_elapsed:.1f}s", flush=True)
        print(f"[{_ts()}]   Examples: {len(examples)}", flush=True)
        
        if not examples:
            print(f"[{_ts()}] ERROR: No examples loaded!", flush=True)
            return 1
        
        # ================================================================
        # PHASE 3: SCORING EXAMPLES
        # ================================================================
        print(f"\n[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] PHASE 3: SCORING EXAMPLES", flush=True)
        print(f"[{_ts()}] {'='*60}", flush=True)
        
        # Process each example
        all_results = []
        all_scores = []
        
        for ex_idx, example in enumerate(examples):
            print(f"\n[{_ts()}] Example {ex_idx + 1}/{len(examples)}: {example.id}", flush=True)
            print(f"[{_ts()}]   Chunks: {example.num_chunks}, Answer length: {len(example.answer_text)} chars", flush=True)
            print(f"[{_ts()}]   Answer text: {repr(example.answer_text)}", flush=True)
            
            try:
                # Align chunks to tokens
                chunk_spans, answer_span, encoding_info = align_chunks_to_tokens(
                    example.full_text,
                    example.chunks,
                    example.answer_start_char,
                    example.answer_end_char,
                    tokenizer,
                    config.max_seq_len,
                )
                
                validate_spans(chunk_spans, answer_span, encoding_info["n_tokens"])
                
                logger.info(f"  Tokenized: {encoding_info['n_tokens']} tokens, "
                           f"{len(chunk_spans)} chunk spans, "
                           f"{answer_span.length} answer tokens")
                
                # Compute chunk scores
                chunk_scores = compute_chunk_scores(
                    model,
                    tokenizer,
                    example,
                    chunk_spans,
                    answer_span,
                    model_info,
                    config,
                )
                
                # Store results
                all_scores.append(chunk_scores)
                
                result = {
                    "example_id": example.id,
                    "n_chunks": example.num_chunks,
                    "n_tokens": encoding_info["n_tokens"],
                    "scores": [asdict(s) for s in chunk_scores],
                }
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"  Error processing example: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compute metrics
        logger.info("\nComputing metrics...")
        metrics = compute_metrics(
            all_scores,
            config.score_method,
            config.enable_shuffled_baseline,
        )
        
        print_metrics_summary(metrics)
        
        # Save artifacts
        artifacts.save_results(all_results)
        artifacts.save_metrics(metrics.to_dict())
        
        elapsed = time.time() - start_time
        
        summary = {
            "mode": args.mode,
            "n_examples_processed": len(all_results),
            "n_examples_requested": config.max_examples,
            "edge_layers": config.edge_layers,
            "edge_heads": config.edge_heads,
            "elapsed_seconds": elapsed,
            "success": True,
        }
        artifacts.save_summary(summary)
        
        logger.info(f"\nRun completed in {elapsed:.1f}s")
        logger.info(f"Artifacts saved to: {run_dir}")
        
        # Print clear PASS/FAIL
        print("\n" + "=" * 60)
        if len(all_results) > 0:
            print(f"✓ {args.mode.upper()} PASS - Processed {len(all_results)} examples")
        else:
            print(f"✗ {args.mode.upper()} FAIL - No examples processed")
        print("=" * 60)
        
        return 0 if len(all_results) > 0 else 1
        
    except Exception as e:
        logger.error(f"Run failed: {e}")
        import traceback
        traceback.print_exc()
        
        summary = {
            "mode": args.mode,
            "success": False,
            "error": str(e),
        }
        artifacts.save_summary(summary)
        
        print("\n" + "=" * 60)
        print(f"✗ {args.mode.upper()} FAIL - {e}")
        print("=" * 60)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
