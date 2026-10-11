from __future__ import annotations

from typing import Any

import numpy as np

from recint.data.categories import ItemCategories
from recint.data.interactions import InteractionData
from recint.interventions.base import Intervention
from recint.models import Recommender, build_model


def _standardize(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-row z-score over valid entries (0 where invalid or constant)."""
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    filled = np.where(valid, scores, 0.0)
    mean = filled.sum(axis=1, keepdims=True) / count
    std = np.sqrt((np.where(valid, scores - mean, 0.0) ** 2).sum(axis=1, keepdims=True) / count)
    return np.divide(filled - mean, std, out=np.zeros_like(filled), where=valid & (std > 0))


class ScoreFusion(Intervention):
    """Blend the base scores with a second recommender fitted on the same history.

    new_score = (1 - weight) * z(base) + weight * z(second), z = per-user z-score over
    unseen items. Unlike the candidate re-rankers it scores the full catalog, so it can
    surface items the base model ranked low: it adds evidence from another model.
    """

    def __init__(self, weight: float, model: dict[str, Any]) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        self.weight = weight
        self.model_config = model
        self._model: Recommender | None = None

    def fit(self, train: InteractionData, item_categories: ItemCategories | None = None) -> ScoreFusion:
        self._model = build_model(self.model_config).fit(train)
        return self

    def apply(self, user_indices: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before apply()")
        valid = np.isfinite(base_scores)
        second = self._model.score(user_indices)
        fused = (1.0 - self.weight) * _standardize(base_scores, valid) + self.weight * _standardize(second, valid)
        return np.where(valid, fused, -np.inf)
