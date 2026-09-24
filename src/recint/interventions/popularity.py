from __future__ import annotations

import numpy as np

from recint.data.interactions import InteractionData
from recint.interventions.base import Intervention
from recint.models.base import top_k_items


def _min_max(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    low = values.min(axis=axis, keepdims=True)
    span = values.max(axis=axis, keepdims=True) - low
    return np.divide(values - low, span, out=np.zeros_like(values), where=span > 0)


class PopularityPenaltyReranker(Intervention):
    """Rerank the base top-`candidate_size` items with a popularity penalty.

    new_score = minmax_user(base_score) - penalty_weight * minmax(log1p(train_popularity))

    Only the base candidates can be recommended after the intervention.
    """

    def __init__(self, penalty_weight: float, candidate_size: int) -> None:
        if penalty_weight < 0:
            raise ValueError("penalty_weight must be >= 0")
        if candidate_size < 1:
            raise ValueError("candidate_size must be >= 1")
        self.penalty_weight = penalty_weight
        self.candidate_size = candidate_size
        self._popularity: np.ndarray | None = None

    def fit(self, train: InteractionData) -> PopularityPenaltyReranker:
        counts = np.bincount(train.interactions["item_idx"].to_numpy(), minlength=train.n_items)
        self._popularity = _min_max(np.log1p(counts.astype(np.float64)))
        return self

    def apply(self, user_indices: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        if self._popularity is None:
            raise RuntimeError("Call fit() before apply()")
        candidates = top_k_items(base_scores, self.candidate_size)
        candidate_scores = np.take_along_axis(base_scores, candidates, axis=1)
        valid = np.isfinite(candidate_scores)

        # Normalize only over finite candidates; masked (seen) items stay -inf.
        low = np.where(valid, candidate_scores, np.inf).min(axis=1, keepdims=True)
        high = np.where(valid, candidate_scores, -np.inf).max(axis=1, keepdims=True)
        span = high - low
        normalized = np.divide(
            np.where(valid, candidate_scores - np.where(np.isfinite(low), low, 0.0), 0.0),
            span,
            out=np.zeros_like(candidate_scores),
            where=valid & (span > 0),
        )
        reranked = normalized - self.penalty_weight * self._popularity[candidates]

        intervened = np.full_like(base_scores, -np.inf)
        np.put_along_axis(intervened, candidates, np.where(valid, reranked, -np.inf), axis=1)
        return intervened
