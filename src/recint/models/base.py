from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from recint.data.interactions import InteractionData


class Recommender(ABC):
    """A base recommender maps users to scores over the full item catalog."""

    @abstractmethod
    def fit(self, train: InteractionData) -> Recommender: ...

    @abstractmethod
    def score(self, user_indices: np.ndarray) -> np.ndarray:
        """Return a (len(user_indices), n_items) float score matrix."""


def mask_seen_items(
    scores: np.ndarray, user_item: sp.csr_matrix, user_indices: np.ndarray
) -> np.ndarray:
    """Set scores of training items to -inf, in place."""
    rows, cols = user_item[user_indices].nonzero()
    scores[rows, cols] = -np.inf
    return scores


def top_k_items(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k item indices per row; ties broken by lower item index."""
    if k > scores.shape[1]:
        raise ValueError(f"k={k} exceeds number of items {scores.shape[1]}")
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]
