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
from edgepatch.spans import align_chunks_to_tokens, validate_spans, AlignmentError
from edgepatch.model import load_model_and_tokenizer
from edgepatch.scoring import compute_chunk_scores, compute_chunk_scores_rollout_light, ChunkScore
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
    elif args.mode == "confirm":
        if args.max_examples is None:
            args.max_examples = 3
        # Set default max_scan_items for confirm
        if not hasattr(args, 'max_scan_items') or getattr(args, 'max_scan_items', None) is None:
            base_config.max_scan_items = 5000
    elif args.mode == "nuclear":
        # Nuclear mode: mask ALL layers to prove causal connection
        if args.max_examples is None:
            args.max_examples = 1
        # Force all layers (32 for DeepSeek-R1-Distill-Llama-8B)
        args.edge_layers = list(range(32))
        args.edge_heads = None  # ALL heads
        if not hasattr(args, 'max_scan_items') or getattr(args, 'max_scan_items', None) is None:
            base_config.max_scan_items = 2000
        print(f"[{_ts()}] NUCLEAR MODE: Masking ALL 32 layers, ALL heads", flush=True)
        print(f"[{_ts()}] Expected: Large negative delta (proves causal connection)", flush=True)
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
        # PHASE 2: DATASET LOADING (streaming)
        # ================================================================
        print(f"\n[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] PHASE 2: DATASET LOADING", flush=True)
        print(f"[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] Loading dataset: {config.dataset_name}...", flush=True)
        
        dataset_start = time.time()
        
        # ================================================================
        # PHASE 3: SCORING EXAMPLES (with resampling for failures)
        # ================================================================
        print(f"\n[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] PHASE 3: SCORING EXAMPLES", flush=True)
        print(f"[{_ts()}] {'='*60}", flush=True)
        
        # Resampling behavior: keep processing until max_examples are SUCCESSFULLY scored
        # Cap failures to prevent infinite loops
        MAX_FAILURES = 50
        RESAMPLING_MODES = {"confirm", "main"}  # Smoke and nuclear don't resample
        
        all_results = []
        all_scores = []
        failed_examples = []
        scanned_count = 0
        
        # Use generator to stream examples
        example_generator = load_dataset_examples(config)
        
        for example in example_generator:
            scanned_count += 1
            
            # Check if we have enough successful examples
            if len(all_results) >= config.max_examples:
                print(f"[{_ts()}] Reached {config.max_examples} successful examples, stopping.", flush=True)
                break
            
            # Check failure cap
            if len(failed_examples) >= MAX_FAILURES:
                raise RuntimeError(
                    f"Exceeded {MAX_FAILURES} alignment failures. "
                    f"Processed {len(all_results)}/{config.max_examples} examples. "
                    f"Aborting to prevent infinite loop."
                )
            
            print(f"\n[{_ts()}] Example {len(all_results) + 1}/{config.max_examples}: {example.id}", flush=True)
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
                
                # Method routing: choose scoring method
                # smoke/nuclear: always use legacy for sanity checks
                # confirm/main: use rollout_light by default (unless --method legacy set)
                use_rollout_light = (
                    config.method == "rollout_light" or 
                    (config.method != "legacy" and args.mode in ["confirm", "main"])
                )
                
                # Force legacy for smoke/nuclear modes
                if args.mode in ["smoke", "nuclear"]:
                    use_rollout_light = False
                
                method_details = None
                
                if use_rollout_light:
                    print(f"[{_ts()}] Using rollout_light scoring method", flush=True)
                    chunk_scores, method_details = compute_chunk_scores_rollout_light(
                        model,
                        tokenizer,
                        example,
                        chunk_spans,
                        answer_span,
                        model_info,
                        config,
                    )
                else:
                    print(f"[{_ts()}] Using legacy (KL) scoring method", flush=True)
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
                    "method": "rollout_light" if use_rollout_light else "legacy",
                    "scores": [asdict(s) for s in chunk_scores],
                }
                
                # Add method details if available
                if method_details:
                    result["method_details"] = method_details
                
                all_results.append(result)
                
                # Incremental sync to Drive (if configured)
                if getattr(args, 'drive_sync_dir', None):
                    drive_sync_path = Path(args.drive_sync_dir)
                    drive_sync_path.mkdir(parents=True, exist_ok=True)
                    sync_file = drive_sync_path / "all_results_incremental.json"
                    with open(sync_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    print(f"[{_ts()}]   ☁️ Synced {len(all_results)} results to Drive", flush=True)
                
            except AlignmentError as e:
                # Alignment failure - log and continue (resample in confirm/main)
                failed_info = {
                    "example_id": example.id,
                    "answer_text": example.answer_text,
                    "char_start": example.answer_start_char,
                    "char_end": example.answer_end_char,
                    "error": str(e),
                    "context_snippet": example.full_text[
                        max(0, example.answer_start_char - 50):
                        min(len(example.full_text), example.answer_end_char + 50)
                    ],
                }
                failed_examples.append(failed_info)
                
                if args.mode in RESAMPLING_MODES:
                    logger.warning(f"  Alignment failed for {example.id}: {e}")
                    logger.warning(f"  Resampling... ({len(failed_examples)}/{MAX_FAILURES} failures)")
                    continue
                else:
                    # Smoke/nuclear: don't resample, just log and continue
                    logger.error(f"  Alignment failed for {example.id}: {e}")
                    continue
                    
            except Exception as e:
                # Other errors - log and continue
                logger.error(f"  Error processing example: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        dataset_elapsed = time.time() - dataset_start
        print(f"\n[{_ts()}] Dataset processing complete in {dataset_elapsed:.1f}s", flush=True)
        print(f"[{_ts()}]   Scanned: {scanned_count}", flush=True)
        print(f"[{_ts()}]   Processed: {len(all_results)}", flush=True)
        print(f"[{_ts()}]   Failed: {len(failed_examples)}", flush=True)
        
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
            "n_examples_requested": config.max_examples,
            "n_examples_processed": len(all_results),
            "n_examples_failed": len(failed_examples),
            "n_examples_scanned": scanned_count,
            "failed_example_ids": [f["example_id"] for f in failed_examples],
            "edge_layers": config.edge_layers,
            "edge_heads": config.edge_heads,
            "elapsed_seconds": elapsed,
            "success": len(all_results) > 0,
        }
        artifacts.save_summary(summary)
        
        # Save failed examples for debugging
        if failed_examples:
            artifacts.save_failed_examples(failed_examples)
            print(f"[{_ts()}] Saved {len(failed_examples)} failed examples to failed_examples.jsonl", flush=True)
        
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
