"""Per-user beyond-accuracy quality of top-K lists; higher is better for every metric.

- tail@K: share of long-tail items (outside the most popular `head_fraction` of items).
- ild@K: intra-list diversity, 1 - mean pairwise cosine similarity of item
  co-occurrence vectors in the history (the similarity MMR uses).
- calibration@K: -KL(p_u || q~_L) over item categories (Steck 2018), p_u from the
  user's history, q~ the list's category distribution smoothed with p_u.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from recint.data.categories import ItemCategories
from recint.data.interactions import InteractionData

BEYOND_ACCURACY_METRICS = ("tail", "ild", "calibration")
MAX_DENSE_SIMILARITY_ITEMS = 20_000
CALIBRATION_SMOOTHING = 0.01


class ListQualityMetrics:
    def __init__(
        self,
        history: InteractionData,
        item_categories: ItemCategories | None,
        metrics: Sequence[str],
        head_fraction: float,
    ) -> None:
        unknown = set(metrics) - set(BEYOND_ACCURACY_METRICS)
        if unknown:
            raise ValueError(f"Unknown beyond-accuracy metrics {sorted(unknown)}; available: {BEYOND_ACCURACY_METRICS}")
        if "calibration" in metrics and item_categories is None:
            raise ValueError("calibration@K needs item categories in the dataset config")
        self.metrics = list(metrics)
        self._user_items = history.user_item_matrix()

        counts = np.asarray(self._user_items.sum(axis=0)).ravel()
        n_head = int(np.ceil(head_fraction * history.n_items))
        head = np.argsort(-counts, kind="stable")[:n_head]
        self._is_tail = np.ones(history.n_items, dtype=bool)
        self._is_tail[head] = False

        self._similarity = None
        if "ild" in self.metrics:
            if history.n_items > MAX_DENSE_SIMILARITY_ITEMS:
                raise ValueError(f"ild@K uses a dense item similarity; catalog exceeds {MAX_DENSE_SIMILARITY_ITEMS} items")
            item_user = self._user_items.T.tocsr()
            norms = np.sqrt(np.asarray(item_user.multiply(item_user).sum(axis=1)).ravel())
            inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
            vectors = sp.diags(inverse) @ item_user
            self._similarity = (vectors @ vectors.T).toarray()
        self._categories = None if item_categories is None else item_categories.distribution.toarray()

    def compute(self, users: np.ndarray, ranked_items: np.ndarray, k_values: Sequence[int]) -> dict[str, np.ndarray]:
        """Values keyed like 'ild@10' for users (rows of `ranked_items`)."""
        results = {}
        for k in k_values:
            top = ranked_items[:, :k]
            if "tail" in self.metrics:
                results[f"tail@{k}"] = self._is_tail[top].mean(axis=1)
            if "ild" in self.metrics:
                pairwise = self._similarity[top[:, :, None], top[:, None, :]]
                off_diagonal = pairwise.sum(axis=(1, 2)) - np.trace(pairwise, axis1=1, axis2=2)
                results[f"ild@{k}"] = 1.0 - off_diagonal / (k * (k - 1))
            if "calibration" in self.metrics:
                results[f"calibration@{k}"] = -self._miscalibration(users, top)
        return results

    def _miscalibration(self, users: np.ndarray, top: np.ndarray) -> np.ndarray:
        history_mass = np.asarray((self._user_items[users] @ self._categories))
        totals = history_mass.sum(axis=1, keepdims=True)
        target = np.divide(history_mass, totals, out=np.zeros_like(history_mass), where=totals > 0)
        listed = self._categories[top].mean(axis=1)
        smoothed = (1.0 - CALIBRATION_SMOOTHING) * listed + CALIBRATION_SMOOTHING * target
        ratio = np.divide(target, smoothed, out=np.ones_like(target), where=target > 0)
        return (target * np.log(ratio)).sum(axis=1)
