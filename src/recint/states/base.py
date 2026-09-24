from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from recint.data.interactions import InteractionData


class UserStateExtractor(ABC):
    """Computes one scalar state value per user from training data only."""

    @abstractmethod
    def compute(self, train: InteractionData) -> np.ndarray:
        """Return an (n_users,) float array indexed by user index."""
