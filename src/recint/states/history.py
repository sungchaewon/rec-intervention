from __future__ import annotations

import numpy as np
from scipy.special import entr

from recint.states.base import StateContext, UserStateExtractor


class HistoryLengthState(UserStateExtractor):
    """Number of training interactions (history sparsity)."""

    def compute(self, context: StateContext) -> np.ndarray:
        return context.train.history_lengths()[context.eval_users].astype(np.float64)


class HistoryPopularityState(UserStateExtractor):
    """Mean popularity of the user's history items, as log1p(count) / log1p(max count) in [0, 1]."""

    def compute(self, context: StateContext) -> np.ndarray:
        train = context.train
        counts = np.bincount(train.interactions["item_idx"].to_numpy(), minlength=train.n_items)
        popularity = np.log1p(counts) / np.log1p(max(counts.max(), 1))
        user_items = train.user_item_matrix()[context.eval_users]
        lengths = np.asarray(user_items.sum(axis=1)).ravel()
        totals = np.asarray(user_items @ popularity).ravel()
        return np.divide(totals, lengths, out=np.zeros_like(totals), where=lengths > 0)


class HistoryCategoryEntropyState(UserStateExtractor):
    """Normalized entropy of the user's history category distribution (0 = one category)."""

    def compute(self, context: StateContext) -> np.ndarray:
        if context.item_categories is None:
            raise ValueError("history_category_entropy needs item categories in the dataset config")
        mass = np.asarray(
            (context.train.user_item_matrix()[context.eval_users] @ context.item_categories.distribution).todense()
        )
        totals = mass.sum(axis=1, keepdims=True)
        distribution = np.divide(mass, totals, out=np.zeros_like(mass), where=totals > 0)
        max_entropy = np.log(max(context.item_categories.n_categories, 2))
        return entr(distribution).sum(axis=1) / max_entropy
