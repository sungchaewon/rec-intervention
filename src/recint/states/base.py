from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from recint.data.interactions import InteractionData


@dataclass(frozen=True)
class StateContext:
    """Everything a user state may depend on. Never includes intervention output,
    so states describe the user/base model before any intervention.

    `base_scores` rows align with `eval_users`; seen items are -inf. Read-only.
    """

    train: InteractionData
    eval_users: np.ndarray
    base_scores: np.ndarray


class UserStateExtractor(ABC):
    """Computes one scalar state value per evaluated user."""

    @abstractmethod
    def compute(self, context: StateContext) -> np.ndarray:
        """Return a (len(context.eval_users),) float array."""
