from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from recint.data.interactions import InteractionData


class Recommender(ABC):
    """A base recommender maps users to scores over the full item catalog."""

    # True if `fit(train)` never trains on each user's last train item (it is
    # held out, e.g. for early stopping). Such a model can produce leakage-free
    # scores for that item from the shortened history; others must be refit.
    holds_out_last_item: bool = False

    @abstractmethod
    def fit(self, train: InteractionData) -> Recommender: ...

    @abstractmethod
    def score(self, user_indices: np.ndarray, history: InteractionData | None = None) -> np.ndarray:
        """Return a (len(user_indices), n_items) float score matrix.

        `history` (same id space) replaces the users' fitted interactions as
        model input, e.g. a history shortened by one item for validation.
        """

    def save(self, path: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoints")

    @classmethod
    def load(cls, path: Path, train: InteractionData, **params: Any) -> Recommender:
        """Restore a fitted model; `train` must be the data it was fitted on."""
        raise NotImplementedError(f"{cls.__name__} does not support checkpoints")


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
