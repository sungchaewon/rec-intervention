"""Per-user top-K ranking metrics with binary relevance.

`targets` is either a 1-D array (one held-out item per user, leave-one-out) or a
sequence of arrays (a set of relevant items per user).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Targets = np.ndarray | Sequence[np.ndarray]


def _relevance(ranked_items: np.ndarray, targets: Targets, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (relevance (n_users, k) bool, n_relevant per user)."""
    if k < 1 or k > ranked_items.shape[1]:
        raise ValueError(f"k={k} must be in [1, {ranked_items.shape[1]}]")
    if len(targets) != len(ranked_items):
        raise ValueError("ranked_items and targets must have the same number of users")
    top = ranked_items[:, :k]
    if isinstance(targets, np.ndarray) and targets.ndim == 1:
        return top == targets[:, None], np.ones(len(top), dtype=np.int64)
    relevance = np.array([np.isin(row, target) for row, target in zip(top, targets)], dtype=bool)
    n_relevant = np.array([len(np.unique(target)) for target in targets], dtype=np.int64)
    return relevance.reshape(top.shape), n_relevant


def hit_rate_at_k(ranked_items: np.ndarray, targets: Targets, k: int) -> np.ndarray:
    """1.0 if any relevant item is in the top-k, else 0.0."""
    relevance, _ = _relevance(ranked_items, targets, k)
    return relevance.any(axis=1).astype(np.float64)


def ndcg_at_k(ranked_items: np.ndarray, targets: Targets, k: int) -> np.ndarray:
    relevance, n_relevant = _relevance(ranked_items, targets, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = relevance @ discounts
    ideal_dcg = np.concatenate([[0.0], np.cumsum(discounts)])[np.minimum(n_relevant, k)]
    return np.divide(dcg, ideal_dcg, out=np.zeros_like(dcg), where=ideal_dcg > 0)


RANKING_METRICS: dict[str, Callable[[np.ndarray, Targets, int], np.ndarray]] = {
    "hr": hit_rate_at_k,
    "ndcg": ndcg_at_k,
}


def metric_name(metric: str, k: int) -> str:
    return f"{metric}@{k}"


def compute_ranking_metrics(
    ranked_items: np.ndarray,
    targets: Targets,
    metrics: Sequence[str],
    k_values: Sequence[int],
) -> dict[str, np.ndarray]:
    """Per-user values keyed by e.g. 'ndcg@10'."""
    unknown = set(metrics) - set(RANKING_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}; available: {sorted(RANKING_METRICS)}")
    return {
        metric_name(metric, k): RANKING_METRICS[metric](ranked_items, targets, k)
        for metric in metrics
        for k in k_values
    }
