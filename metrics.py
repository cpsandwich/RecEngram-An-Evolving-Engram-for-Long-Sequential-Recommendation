"""Standard top-K ranking metrics: Recall@K, NDCG@K, MRR."""

import numpy as np
from typing import Dict


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


def mrr_at_k(ranks: np.ndarray, k: int) -> float:
    """Mean Reciprocal Rank @K."""
    if len(ranks) == 0:
        return 0.0
    reciprocal = np.where(ranks <= k, 1.0 / ranks, 0.0)
    return float(reciprocal.mean())


def compute_all_metrics(scores: np.ndarray, k: int = 10) -> Dict[str, float]:
    """Compute Recall@K, NDCG@K, MRR for a single batch or full set.

    Args:
        scores: (B, N_items) prediction logits, col 0 = positive
        k: cutoff
    """
    ranks = _compute_ranks(scores)
    return {
        f"Recall@{k}": recall_at_k(ranks, k),
        f"NDCG@{k}": ndcg_at_k(ranks, k),
        f"MRR@{k}": mrr_at_k(ranks, k),
    }
