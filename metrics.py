"""Standard top-K ranking metrics: Recall@K, NDCG@K, HR@K, MRR."""

import numpy as np
from typing import Dict, List


def _compute_ranks(scores: np.ndarray) -> np.ndarray:
    """Compute 1-indexed ranks where column 0 is the positive item.

    scores: (B, N_items) — higher score = better rank
    Returns: (B,) 1-indexed ranks of positive items
    """
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]
    B = scores.shape[0]
    pos_scores = scores[:, 0]
    ranks = (scores > pos_scores[:, np.newaxis]).sum(axis=1) + 1  # (B,)
    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Proportion of relevant items in top-K."""
    if len(ranks) == 0:
        return 0.0
    return float((ranks <= k).mean())


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """Normalized DCG@K."""
    if len(ranks) == 0:
        return 0.0
    hits = ranks <= k
    if not hits.any():
        return 0.0
    dcg = np.sum(1.0 / np.log2(ranks[hits] + 1.0))
    idcg = hits.sum() / np.log2(2.0)  # ideal: all hits at rank 1
    return float(dcg / idcg)


def hit_rate_at_k(ranks: np.ndarray, k: int) -> float:
    """Hit Rate @K: proportion of users with at least one hit in top-K."""
    if len(ranks) == 0:
        return 0.0
    return float((ranks <= k).mean())


def mrr_at_k(ranks: np.ndarray, k: int = None) -> float:
    """Mean Reciprocal Rank @K. If k is None, MRR over all ranks."""
    if len(ranks) == 0:
        return 0.0
    if k is not None:
        reciprocal = np.where(ranks <= k, 1.0 / ranks, 0.0)
    else:
        reciprocal = 1.0 / ranks
    return float(reciprocal.mean())


def compute_all_metrics(
    scores: np.ndarray, ks=None, k: int = None
) -> Dict[str, float]:
    """Compute Recall@K, NDCG@K, HR@K, MRR for given K values.

    Args:
        scores: (B, N_items) prediction logits, col 0 = positive
        ks: list of cutoff values (default [5, 10, 20])
        k: single cutoff (backward-compatible, deprecated)

    Returns:
        dict with metric_name -> value
    """
    if ks is None:
        if k is not None:
            ks = [k]
        else:
            ks = [5, 10, 20]
    ranks = _compute_ranks(scores)
    metrics = {}
    for cutoff in ks:
        metrics[f"Recall@{cutoff}"] = recall_at_k(ranks, cutoff)
        metrics[f"NDCG@{cutoff}"] = ndcg_at_k(ranks, cutoff)
        metrics[f"HR@{cutoff}"] = hit_rate_at_k(ranks, cutoff)
    metrics["MRR"] = mrr_at_k(ranks, None)
    return metrics


def compute_metrics_at_k(scores: np.ndarray, k: int = 10) -> Dict[str, float]:
    """Compute metrics at a single K (backward-compatible).

    Args:
        scores: (B, N_items) prediction logits, col 0 = positive
        k: cutoff
    """
    ranks = _compute_ranks(scores)
    return {
        f"Recall@{k}": recall_at_k(ranks, k),
        f"NDCG@{k}": ndcg_at_k(ranks, k),
        f"HR@{k}": hit_rate_at_k(ranks, k),
        f"MRR@{k}": mrr_at_k(ranks, k),
    }
