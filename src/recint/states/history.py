from __future__ import annotations

import numpy as np

from recint.data.interactions import InteractionData
from recint.states.base import UserStateExtractor


class HistoryLengthState(UserStateExtractor):
    """Number of training interactions (history sparsity)."""

    def compute(self, train: InteractionData) -> np.ndarray:
        return train.history_lengths().astype(np.float64)
