from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from recint.data.categories import ItemCategories
from recint.data.interactions import InteractionData
from recint.models.base import top_k_items


class Intervention(ABC):
    """Transforms base recommender scores into intervened scores.

    The base model is never modified: `apply` receives the base scores (with
    seen items already masked to -inf) and returns a new score matrix of the same
    shape without modifying its input. Items that should not be recommended get -inf.
    """

    def fit(self, train: InteractionData, item_categories: ItemCategories | None = None) -> Intervention:
        return self

    @abstractmethod
    def apply(self, user_indices: np.ndarray, base_scores: np.ndarray) -> np.ndarray: ...


class IdentityIntervention(Intervention):
    """No-op control arm; its gain must be exactly zero for every user."""

    def apply(self, user_indices: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        return base_scores.copy()


@dataclass(frozen=True)
class Candidates:
    """The base top-`candidate_size` items per user, in base rank order."""

    items: np.ndarray  # (n_users, candidate_size) item indices
    valid: np.ndarray  # False where the base score is -inf (seen / unavailable)
    relevance: np.ndarray  # base scores min-max normalized per user over valid candidates


def base_candidates(base_scores: np.ndarray, candidate_size: int) -> Candidates:
    items = top_k_items(base_scores, candidate_size)
    scores = np.take_along_axis(base_scores, items, axis=1)
    valid = np.isfinite(scores)
    low = np.where(valid, scores, np.inf).min(axis=1, keepdims=True)
    high = np.where(valid, scores, -np.inf).max(axis=1, keepdims=True)
    span = high - low
    relevance = np.divide(
        np.where(valid, scores - np.where(np.isfinite(low), low, 0.0), 0.0),
        span,
        out=np.zeros_like(scores),
        where=valid & (span > 0),
    )
    return Candidates(items, valid, relevance)


def scores_from_rankings(shape: tuple[int, int], rankings: list[np.ndarray]) -> np.ndarray:
    """Turn per-user ordered item lists into a score matrix (first = highest, others -inf)."""
    scores = np.full(shape, -np.inf)
    for row, ranking in enumerate(rankings):
        scores[row, ranking] = np.arange(len(ranking), 0, -1, dtype=np.float64)
    return scores
