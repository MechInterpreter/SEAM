"""
Configuration dataclass for EdgePatch experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml
import json
import argparse


@dataclass
class EdgePatchConfig:
    """Configuration for Edge-Patch experiments."""
    
    # Dataset
    dataset_name: str = "uzaymacar/math-rollouts"
    dataset_split: str = "default"  # This dataset uses 'default' not 'train'
    max_examples: int = 10
    
    # Streaming & filtering (NEW)
    dataset_streaming: bool = True           # Use streaming=True to avoid full materialization
    force_materialize_dataset: bool = False  # Debug flag to force full load
    example_id_allowlist: Optional[list[str]] = None  # Filter to specific example IDs
    example_id_denylist: Optional[list[str]] = None   # Exclude specific example IDs (e.g., already processed)
    ta_labeled_only: bool = True             # Skip examples without TA labels
    max_scan_items: Optional[int] = None     # Max items to scan in streaming mode (None=unlimited)
    solution_type: str = "correct_base_solution" # Subdirectory to load (e.g. correct_base_solution)
    
    # Model
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    load_in_4bit: bool = True
    max_seq_len: int = 5000
    
    # Edge masking - CRITICAL: must actually control behavior
    # None = all layers/heads; list = only those indices
    edge_layers: Optional[list[int]] = None
    edge_heads: Optional[list[int]] = None
    
    # Scoring
    score_method: str = "delta_logp"  # "delta_logp" or "abs_delta_logp"
    score_span: str = "extended"      # "answer_only", "extended", "reasoning_only"
    score_extend_tokens: int = 20     # Tokens before answer to include when score_span="extended"
    saturation_threshold: float = 0.999  # Warn if all baseline probs exceed this
    
    # TA labels
    ta_label_field: str = "counterfactual_importance_accuracy"
    
    # Output
    output_dir: str = "runs"
    run_name: Optional[str] = None
    
    # Sanity checks
    enable_shuffled_baseline: bool = True
    enable_random_span_control: bool = False  # Expensive, off by default
    
    # Verbosity
    verbose: bool = True
    probe_chunk_0: bool = True  # Print detailed info for first chunk
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_yaml(cls, path: str) -> "EdgePatchConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> "EdgePatchConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace, base_config: Optional["EdgePatchConfig"] = None) -> "EdgePatchConfig":
        """Create config from argparse namespace, optionally overriding a base config."""
        if base_config is not None:
            config_dict = base_config.to_dict()
        else:
            config_dict = {}
        
        # Override with non-None args
        for key, value in vars(args).items():
            if value is not None and key in cls.__dataclass_fields__:
                config_dict[key] = value
        
        # Handle special CLI flags that map to different config names
        if getattr(args, 'no_streaming', False):
            config_dict['dataset_streaming'] = False
        if getattr(args, 'example_ids', None):
            config_dict['example_id_allowlist'] = args.example_ids
        if getattr(args, 'exclude_ids', None):
            config_dict['example_id_denylist'] = args.exclude_ids
        if getattr(args, 'include_unlabeled', False):
            config_dict['ta_labeled_only'] = False
        
        return cls(**config_dict)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser with all config options."""
    parser = argparse.ArgumentParser(
        description="EdgePatch: Causal Receiver Masking Experiments"
    )
    
    # Mode
    parser.add_argument(
        "mode",
        choices=["smoke", "confirm", "main", "nuclear"],
        help="Run mode: smoke (1 example), confirm (3 examples), main (full), nuclear (ALL layers)"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file"
    )
    
    # Dataset
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-split", type=str)
    parser.add_argument("--max-examples", type=int)
    
    # Streaming & filtering
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (forces full dataset materialization)"
    )
    parser.add_argument(
        "--example-ids",
        type=str,
        nargs="+",
        help="Filter to specific example IDs"
    )
    parser.add_argument(
        "--exclude-ids",
        type=str,
        nargs="+",
        help="Exclude specific example IDs (e.g., already processed)"
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include examples without TA labels"
    )
    
    # Model
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--load-in-4bit", type=bool)
    parser.add_argument("--max-seq-len", type=int)
    
    # Edge masking - CRITICAL
    parser.add_argument(
        "--edge-layers",
        type=int,
        nargs="+",
        help="Layer indices to apply masking (default: all)"
    )
    parser.add_argument(
        "--edge-heads",
        type=int,
        nargs="+",
        help="Head indices to apply masking (default: all)"
    )
    
    # Scoring
    parser.add_argument("--score-method", type=str, choices=["delta_logp", "abs_delta_logp"])
    parser.add_argument(
        "--score-span",
        type=str,
        choices=["answer_only", "extended", "reasoning_only"],
        help="What to score: answer_only, extended (answer + N reasoning tokens), reasoning_only"
    )
    parser.add_argument(
        "--score-extend-tokens",
        type=int,
        help="Number of reasoning tokens to include when score_span=extended (default: 20)"
    )
    
    # Output
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--run-name", type=str)
    
    # Flags
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-verbose", action="store_false", dest="verbose")
    
    return parser


def get_config_from_cli() -> EdgePatchConfig:
    """Parse CLI arguments and return config."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Start with base config
    if args.config:
        base_config = EdgePatchConfig.from_yaml(args.config)
    else:
        base_config = EdgePatchConfig()
    
    # Apply mode-specific defaults
    if args.mode == "smoke":
        if args.max_examples is None:
            args.max_examples = 1
    elif args.mode == "confirm":
        if args.max_examples is None:
            args.max_examples = 3
    
    # Override with CLI args
    config = EdgePatchConfig.from_args(args, base_config)
    
    return config
