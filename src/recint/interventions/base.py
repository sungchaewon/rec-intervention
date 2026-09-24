from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from recint.data.interactions import InteractionData


class Intervention(ABC):
    """Transforms base recommender scores into intervened scores.

    The base model is never modified: `apply` receives the base scores (with
    seen items already masked to -inf) and returns a new score matrix of the same
    shape. Items that should not be recommended get -inf.
    """

    def fit(self, train: InteractionData) -> Intervention:
        return self

    @abstractmethod
    def apply(self, user_indices: np.ndarray, base_scores: np.ndarray) -> np.ndarray: ...


class IdentityIntervention(Intervention):
    """No-op control arm; its gain must be exactly zero for every user."""

    def apply(self, user_indices: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        return base_scores.copy()
