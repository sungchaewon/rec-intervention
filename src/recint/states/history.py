from __future__ import annotations

import numpy as np

from recint.states.base import StateContext, UserStateExtractor


class HistoryLengthState(UserStateExtractor):
    """Number of training interactions (history sparsity)."""

    def compute(self, context: StateContext) -> np.ndarray:
        return context.train.history_lengths()[context.eval_users].astype(np.float64)
