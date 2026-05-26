"""Statistical significance testing via paired bootstrap resampling.

Paper: 10,000 resamples, users resampled with replacement,
one-sided p-value = fraction of iterations where baseline >= RecEngram.
Also computes 95% bootstrap confidence intervals.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def paired_bootstrap_test(
    model_a_scores: np.ndarray,
    model_b_scores: np.ndarray,
    n_resamples: int = 10000,
    k: int = 10,
    seed: int = 42,
) -> Dict:
    """Paired bootstrap test comparing two models' Recall@K.

    For each bootstrap iteration:
    1. Resample users with replacement
    2. Compute Recall@K for both models on the resampled set
    3. Count how often baseline (model_b) >= RecEngram (model_a)

    Paper: one-sided p-value = fraction where baseline >= RecEngram.
    p < 0.01 confirms statistical significance.

    Args:
        model_a_scores: (N_users, 1+N_items) prediction scores — RecEngram
        model_b_scores: (N_users, 1+N_items) prediction scores — baseline (CL4SRec)
        n_resamples: number of bootstrap iterations (default 10000)
        k: cutoff for Recall@K
        seed: random seed for reproducibility

    Returns:
        dict with p_value, mean_diff, ci_lower, ci_upper, baseline_mean, recengram_mean
    """
    rng = np.random.RandomState(seed)
    n_users = model_a_scores.shape[0]

    # Per-user Recall@K
    a_recall = _per_user_recall(model_a_scores, k)  # (N,)
    b_recall = _per_user_recall(model_b_scores, k)  # (N,)

    # Observed difference
    obs_diff = a_recall.mean() - b_recall.mean()

    # Bootstrap
    boot_diffs = np.zeros(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n_users, size=n_users)
        boot_a = a_recall[idx].mean()
        boot_b = b_recall[idx].mean()
        boot_diffs[i] = boot_a - boot_b

    # One-sided p-value: fraction where baseline >= RecEngram (i.e., diff <= 0)
    p_value = (boot_diffs <= 0).mean()

    # 95% confidence interval for the difference
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    return {
        "p_value": float(p_value),
        "significant": p_value < 0.01,
        "obs_diff": float(obs_diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "model_a_mean": float(a_recall.mean()),
        "model_b_mean": float(b_recall.mean()),
        "n_resamples": n_resamples,
        "n_users": n_users,
    }


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_resamples: int = 10000,
    k: int = 10,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict:
    """Bootstrap 95% confidence interval for a single model's Recall@K.

    Args:
        scores: (N_users, 1+N_items) prediction scores
        n_resamples: number of bootstrap iterations
        k: cutoff for Recall@K
        seed: random seed
        alpha: significance level (default 0.05 for 95% CI)

    Returns:
        dict with mean, std, ci_lower, ci_upper
    """
    rng = np.random.RandomState(seed)
    n_users = scores.shape[0]

    per_user = _per_user_recall(scores, k)

    boot_means = np.zeros(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n_users, size=n_users)
        boot_means[i] = per_user[idx].mean()

    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(per_user.mean()),
        "std": float(per_user.std(ddof=1)),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_resamples": n_resamples,
        "n_users": n_users,
    }


def _per_user_recall(scores: np.ndarray, k: int) -> np.ndarray:
    """Compute per-user Recall@K.

    Args:
        scores: (N, M) — col 0 = positive item, cols 1: = negatives
        k: cutoff

    Returns:
        (N,) per-user Recall@K (0.0 or 1.0 for single positive)
    """
    pos_scores = scores[:, 0:1]  # (N, 1)
    ranks = (scores > pos_scores).sum(axis=1) + 1  # (N,)
    return (ranks <= k).astype(np.float32)


def _per_user_ndcg(scores: np.ndarray, k: int) -> np.ndarray:
    """Compute per-user NDCG@K.

    Args:
        scores: (N, M) — col 0 = positive item
        k: cutoff

    Returns:
        (N,) per-user NDCG@K
    """
    pos_scores = scores[:, 0:1]
    ranks = (scores > pos_scores).sum(axis=1) + 1  # (N,)
    hit = ranks <= k
    dcg = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
    idcg = 1.0 / np.log2(2.0)  # ideal: rank 1
    return np.where(hit, dcg / idcg, 0.0).astype(np.float32)


def run_significance_suite(
    recengram_scores: np.ndarray,
    baseline_scores_dict: Dict[str, np.ndarray],
    n_resamples: int = 10000,
    k: int = 10,
    seed: int = 42,
) -> Dict:
    """Run significance tests for RecEngram against all baselines.

    Args:
        recengram_scores: (N, M) RecEngram scores
        baseline_scores_dict: {model_name: (N, M) scores}
        n_resamples: bootstrap iterations
        k: cutoff
        seed: random seed

    Returns:
        {model_name: test_result_dict}
    """
    results = {}
    for name, baseline_scores in baseline_scores_dict.items():
        results[name] = paired_bootstrap_test(
            recengram_scores, baseline_scores,
            n_resamples=n_resamples, k=k, seed=seed,
        )
    return results


def collect_per_user_scores(
    model,
    dataloader,
    device: str = "cuda",
    max_batches: int = None,
) -> np.ndarray:
    """Collect per-user prediction scores from a model.

    Args:
        model: nn.Module
        dataloader: DataLoader
        device: torch device
        max_batches: optional limit

    Returns:
        (N, 1+N_neg) numpy array of prediction scores
    """
    import torch

    model.eval()
    all_scores = []

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.no_grad():
            output = model(batch)
            if isinstance(output, tuple):
                output = output[0]
        all_scores.append(output.cpu().numpy())

    return np.concatenate(all_scores, axis=0)
