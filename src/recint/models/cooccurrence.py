"""Item co-occurrence recommender: a training-free base model for the pipeline.

score(u, j) = sum_{i in history(u)} sim(i, j), with sim = raw co-occurrence
counts or cosine-normalized co-occurrence. Uses a dense item-item matrix, so it
is meant for small catalogs only.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recint.data.interactions import InteractionData
from recint.models.base import Recommender

SIMILARITIES = ("count", "cosine")


class ItemCooccurrenceRecommender(Recommender):
    def __init__(self, similarity: str = "cosine") -> None:
        if similarity not in SIMILARITIES:
            raise ValueError(f"similarity must be one of {SIMILARITIES}, got {similarity!r}")
        self.similarity = similarity
        self._user_items: sp.csr_matrix | None = None
        self._item_similarity: np.ndarray | None = None

    def fit(self, train: InteractionData) -> ItemCooccurrenceRecommender:
        user_items = train.user_item_matrix()
        cooccurrence = (user_items.T @ user_items).toarray()
        np.fill_diagonal(cooccurrence, 0.0)
        if self.similarity == "cosine":
            degree = np.asarray(user_items.sum(axis=0)).ravel()
            norm = np.sqrt(np.outer(degree, degree))
            cooccurrence = np.divide(
                cooccurrence, norm, out=np.zeros_like(cooccurrence), where=norm > 0
            )
        self._user_items = user_items
        self._item_similarity = cooccurrence
        return self

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        if self._user_items is None or self._item_similarity is None:
            raise RuntimeError("Call fit() before score()")
        return np.asarray(self._user_items[user_indices] @ self._item_similarity)
