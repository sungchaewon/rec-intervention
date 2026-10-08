from __future__ import annotations

import numpy as np

from recint.data.categories import ItemCategories
from recint.data.interactions import InteractionData
from recint.interventions.base import Intervention, base_candidates


def _min_max(values: np.ndarray) -> np.ndarray:
    low, span = values.min(), values.max() - values.min()
    return (values - low) / span if span > 0 else np.zeros_like(values)


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

    def fit(
        self, train: InteractionData, item_categories: ItemCategories | None = None
    ) -> PopularityPenaltyReranker:
        counts = np.bincount(train.interactions["item_idx"].to_numpy(), minlength=train.n_items)
        self._popularity = _min_max(np.log1p(counts.astype(np.float64)))
        return self

    def apply(self, user_indices: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        if self._popularity is None:
            raise RuntimeError("Call fit() before apply()")
        candidates = base_candidates(base_scores, self.candidate_size)
        reranked = candidates.relevance - self.penalty_weight * self._popularity[candidates.items]
        intervened = np.full_like(base_scores, -np.inf)
        np.put_along_axis(
            intervened, candidates.items, np.where(candidates.valid, reranked, -np.inf), axis=1
        )
        return intervened
