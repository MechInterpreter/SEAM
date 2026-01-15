"""
Utility functions for EdgePatch.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("edgepatch")


def get_run_dir(output_dir: str, run_name: str | None = None) -> Path:
    """
    Get the run directory path.
    
    If run_name is provided, use it. Otherwise, generate a timestamped name.
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"edgepatch_{timestamp}"
    
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data: Any, path: str | Path) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_layers_heads(layers: list[int] | None, heads: list[int] | None) -> str:
    """Format layers and heads for logging."""
    layers_str = "all" if layers is None else str(sorted(layers))
    heads_str = "all" if heads is None else str(sorted(heads))
    return f"layers={layers_str}, heads={heads_str}"


class RunArtifacts:
    """Manager for run artifacts."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.config_path = run_dir / "config.json"
        self.summary_path = run_dir / "run_summary.json"
        self.results_path = run_dir / "all_results.json"
        self.metrics_path = run_dir / "eval_metrics.json"
        self.log_path = run_dir / "logs.txt"
    
    def save_config(self, config: dict) -> None:
        """Save configuration."""
        save_json(config, self.config_path)
    
    def save_summary(self, summary: dict) -> None:
        """Save run summary."""
        save_json(summary, self.summary_path)
    
    def save_results(self, results: list[dict]) -> None:
        """Save all per-example results."""
        save_json(results, self.results_path)
    
    def save_metrics(self, metrics: dict) -> None:
        """Save evaluation metrics."""
        save_json(metrics, self.metrics_path)
    
    def exists(self) -> bool:
        """Check if artifacts exist."""
        return self.metrics_path.exists()
    
    def load_results(self) -> list[dict]:
        """Load results from file."""
        return load_json(self.results_path)
    
    def load_metrics(self) -> dict:
        """Load metrics from file."""
        return load_json(self.metrics_path)
