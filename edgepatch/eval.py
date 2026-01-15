"""
Evaluation metrics for EdgePatch.

Computes correlation and agreement metrics between predicted scores and TA labels.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import random

import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_curve, auc

from edgepatch.scoring import ChunkScore, get_scores_array, get_ta_labels_array

logger = logging.getLogger("edgepatch")


@dataclass
class EvalMetrics:
    """Evaluation metrics for a run."""
    # Correlation
    spearman_rho: float
    spearman_p: float
    
    # Top-k overlap
    top_1_overlap: float
    top_3_overlap: float
    top_5_overlap: float
    
    # PR-AUC at thresholds
    pr_auc_10: float  # Top 10% as positive
    pr_auc_20: float  # Top 20% as positive
    
    # Sanity checks
    shuffled_rho: Optional[float] = None  # Should be ~0
    
    # Counts
    n_examples: int = 0
    n_chunks_total: int = 0
    
    def to_dict(self) -> dict:
        return {
            "spearman_rho": self.spearman_rho,
            "spearman_p": self.spearman_p,
            "top_1_overlap": self.top_1_overlap,
            "top_3_overlap": self.top_3_overlap,
            "top_5_overlap": self.top_5_overlap,
            "pr_auc_10": self.pr_auc_10,
            "pr_auc_20": self.pr_auc_20,
            "shuffled_rho": self.shuffled_rho,
            "n_examples": self.n_examples,
            "n_chunks_total": self.n_chunks_total,
        }


def compute_metrics(
    all_scores: list[list[ChunkScore]],
    score_method: str = "delta_logp",
    enable_shuffled_baseline: bool = True,
) -> EvalMetrics:
    """
    Compute evaluation metrics across all examples.
    
    Args:
        all_scores: List of per-example ChunkScore lists
        score_method: "delta_logp" or "abs_delta_logp"
        enable_shuffled_baseline: If True, compute shuffled-label baseline
    
    Returns:
        EvalMetrics with all metrics
    """
    # Flatten all scores and labels for aggregate metrics
    all_pred_scores = []
    all_ta_labels = []
    
    for chunk_scores in all_scores:
        pred = get_scores_array(chunk_scores, score_method)
        ta = get_ta_labels_array(chunk_scores)
        all_pred_scores.extend(pred)
        all_ta_labels.extend(ta)
    
    pred_array = np.array(all_pred_scores)
    ta_array = np.array(all_ta_labels)
    
    n_chunks = len(pred_array)
    n_examples = len(all_scores)
    
    if n_chunks < 2:
        logger.warning("Not enough chunks for meaningful metrics")
        return EvalMetrics(
            spearman_rho=0.0,
            spearman_p=1.0,
            top_1_overlap=0.0,
            top_3_overlap=0.0,
            top_5_overlap=0.0,
            pr_auc_10=0.0,
            pr_auc_20=0.0,
            n_examples=n_examples,
            n_chunks_total=n_chunks,
        )
    
    # Spearman correlation
    spearman_result = stats.spearmanr(pred_array, ta_array)
    spearman_rho = float(spearman_result.correlation)
    spearman_p = float(spearman_result.pvalue)
    
    # Handle NaN
    if np.isnan(spearman_rho):
        spearman_rho = 0.0
        spearman_p = 1.0
    
    # Top-k overlap (per example, then average)
    top_1_overlaps = []
    top_3_overlaps = []
    top_5_overlaps = []
    
    for chunk_scores in all_scores:
        if len(chunk_scores) < 1:
            continue
        
        pred = get_scores_array(chunk_scores, score_method)
        ta = get_ta_labels_array(chunk_scores)
        
        pred_ranks = np.argsort(pred)[::-1]  # Descending
        ta_ranks = np.argsort(ta)[::-1]
        
        # Top-1
        top_1_overlaps.append(1.0 if pred_ranks[0] == ta_ranks[0] else 0.0)
        
        # Top-3
        k3 = min(3, len(pred))
        pred_top3 = set(pred_ranks[:k3])
        ta_top3 = set(ta_ranks[:k3])
        top_3_overlaps.append(len(pred_top3 & ta_top3) / k3)
        
        # Top-5
        k5 = min(5, len(pred))
        pred_top5 = set(pred_ranks[:k5])
        ta_top5 = set(ta_ranks[:k5])
        top_5_overlaps.append(len(pred_top5 & ta_top5) / k5)
    
    top_1_overlap = float(np.mean(top_1_overlaps)) if top_1_overlaps else 0.0
    top_3_overlap = float(np.mean(top_3_overlaps)) if top_3_overlaps else 0.0
    top_5_overlap = float(np.mean(top_5_overlaps)) if top_5_overlaps else 0.0
    
    # PR-AUC at thresholds
    pr_auc_10 = compute_pr_auc_at_threshold(pred_array, ta_array, 0.10)
    pr_auc_20 = compute_pr_auc_at_threshold(pred_array, ta_array, 0.20)
    
    # Shuffled baseline
    shuffled_rho = None
    if enable_shuffled_baseline:
        shuffled_ta = ta_array.copy()
        np.random.shuffle(shuffled_ta)
        shuffled_result = stats.spearmanr(pred_array, shuffled_ta)
        shuffled_rho = float(shuffled_result.correlation)
        if np.isnan(shuffled_rho):
            shuffled_rho = 0.0
    
    return EvalMetrics(
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        top_1_overlap=top_1_overlap,
        top_3_overlap=top_3_overlap,
        top_5_overlap=top_5_overlap,
        pr_auc_10=pr_auc_10,
        pr_auc_20=pr_auc_20,
        shuffled_rho=shuffled_rho,
        n_examples=n_examples,
        n_chunks_total=n_chunks,
    )


def compute_pr_auc_at_threshold(
    pred_scores: np.ndarray,
    ta_labels: np.ndarray,
    threshold_pct: float,
) -> float:
    """
    Compute PR-AUC treating top threshold_pct of TA labels as positive.
    
    Args:
        pred_scores: Predicted importance scores
        ta_labels: Ground truth TA labels
        threshold_pct: Fraction of top TA labels to treat as positive (0.10 = top 10%)
    
    Returns:
        PR-AUC score
    """
    n = len(ta_labels)
    k = max(1, int(n * threshold_pct))
    
    # Binary labels: top-k TA labels are positive
    ta_ranks = np.argsort(ta_labels)[::-1]
    binary_labels = np.zeros(n)
    binary_labels[ta_ranks[:k]] = 1
    
    if binary_labels.sum() == 0 or binary_labels.sum() == n:
        return 0.0
    
    try:
        precision, recall, _ = precision_recall_curve(binary_labels, pred_scores)
        return float(auc(recall, precision))
    except Exception:
        return 0.0


def print_metrics_summary(metrics: EvalMetrics) -> None:
    """Print a summary of metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Examples: {metrics.n_examples}, Chunks: {metrics.n_chunks_total}")
    print("-" * 60)
    print(f"Spearman ρ:     {metrics.spearman_rho:.4f} (p={metrics.spearman_p:.4e})")
    print(f"Top-1 overlap:  {metrics.top_1_overlap:.4f}")
    print(f"Top-3 overlap:  {metrics.top_3_overlap:.4f}")
    print(f"Top-5 overlap:  {metrics.top_5_overlap:.4f}")
    print(f"PR-AUC@10%:     {metrics.pr_auc_10:.4f}")
    print(f"PR-AUC@20%:     {metrics.pr_auc_20:.4f}")
    if metrics.shuffled_rho is not None:
        print(f"Shuffled ρ:     {metrics.shuffled_rho:.4f} (should be ~0)")
    print("=" * 60)
