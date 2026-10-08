"""Beyond-accuracy re-rankers over the base top-`candidate_size` items."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recint.data.categories import ItemCategories
from recint.data.interactions import InteractionData
from recint.interventions.base import Intervention, base_candidates, scores_from_rankings


def _check_weight(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


class MMRReranker(Intervention):
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    Greedily picks argmax relevance_weight * rel(i) - (1 - relevance_weight) * max_{j picked} sim(i, j),
    where rel is the per-user min-max normalized base score and sim is the cosine
    similarity of item co-occurrence vectors in train (independent of the base model).
    """

    def __init__(self, relevance_weight: float, candidate_size: int) -> None:
        _check_weight("relevance_weight", relevance_weight)
        self.relevance_weight = relevance_weight
        self.candidate_size = candidate_size
        self._item_vectors: sp.csr_matrix | None = None

    def fit(self, train: InteractionData, item_categories: ItemCategories | None = None) -> MMRReranker:
        item_user = train.user_item_matrix().T.tocsr()
        norms = np.sqrt(np.asarray(item_user.multiply(item_user).sum(axis=1)).ravel())
        inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        self._item_vectors = (sp.diags(inverse) @ item_user).tocsr()
        return self

    def apply(self, user_indices: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        if self._item_vectors is None:
            raise RuntimeError("Call fit() before apply()")
        candidates = base_candidates(base_scores, self.candidate_size)
        rankings = []
        for items, valid, relevance in zip(candidates.items, candidates.valid, candidates.relevance):
            items, relevance = items[valid], relevance[valid]
            vectors = self._item_vectors[items]
            similarity = (vectors @ vectors.T).toarray()
            rankings.append(items[self._greedy_order(relevance, similarity)])
        return scores_from_rankings(base_scores.shape, rankings)

    def _greedy_order(self, relevance: np.ndarray, similarity: np.ndarray) -> np.ndarray:
        n = len(relevance)
        max_similarity = np.zeros(n)
        picked = np.zeros(n, dtype=bool)
        order = np.empty(n, dtype=np.int64)
        for step in range(n):
            objective = self.relevance_weight * relevance - (1.0 - self.relevance_weight) * max_similarity
            objective[picked] = -np.inf
            choice = int(np.argmax(objective))  # ties -> better base rank
            order[step], picked[choice] = choice, True
            max_similarity = np.maximum(max_similarity, similarity[choice])
        return order


class CalibrationReranker(Intervention):
    """Calibrated recommendations (Steck, RecSys 2018) over item categories.

    Greedily builds the first `list_size` positions maximizing
    (1 - calibration_weight) * sum rel(i) - calibration_weight * KL(p_u || q~_L),
    with p_u the category distribution of the user's train history, q_L the mean
    category distribution of the list, and q~ = (1 - smoothing) q_L + smoothing p_u.
    Remaining candidates follow in base order.
    """

    def __init__(
        self, calibration_weight: float, candidate_size: int, list_size: int, smoothing: float = 0.01
    ) -> None:
        _check_weight("calibration_weight", calibration_weight)
        if not 0.0 < smoothing < 1.0:
            raise ValueError("smoothing must be in (0, 1)")
        self.calibration_weight = calibration_weight
        self.candidate_size = candidate_size
        self.list_size = list_size
        self.smoothing = smoothing
        self._item_distribution: sp.csr_matrix | None = None
        self._user_items: sp.csr_matrix | None = None

    def fit(
        self, train: InteractionData, item_categories: ItemCategories | None = None
    ) -> CalibrationReranker:
        if item_categories is None:
            raise ValueError("calibration needs item categories (set `item_categories_path` in the dataset config)")
        self._item_distribution = item_categories.distribution.tocsr()
        self._user_items = train.user_item_matrix()
        return self

    def apply(self, user_indices: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        if self._item_distribution is None or self._user_items is None:
            raise RuntimeError("Call fit() before apply()")
        candidates = base_candidates(base_scores, self.candidate_size)
        history_categories = self._user_items[user_indices] @ self._item_distribution
        rankings = []
        for row, (items, valid, relevance) in enumerate(
            zip(candidates.items, candidates.valid, candidates.relevance)
        ):
            items, relevance = items[valid], relevance[valid]
            target = history_categories[row].toarray().ravel()
            if target.sum() == 0:  # no category information for this user: keep base order
                rankings.append(items)
                continue
            order = self._greedy_order(
                relevance, self._item_distribution[items].toarray(), target / target.sum()
            )
            rankings.append(items[order])
        return scores_from_rankings(base_scores.shape, rankings)

    def _greedy_order(self, relevance: np.ndarray, distributions: np.ndarray, target: np.ndarray) -> np.ndarray:
        n = len(relevance)
        picked = np.zeros(n, dtype=bool)
        order: list[int] = []
        list_categories = np.zeros_like(target)
        support = target > 0
        for size in range(1, min(self.list_size, n) + 1):
            # Category distribution of the list if each candidate were appended.
            list_distribution = (list_categories + distributions) / size
            smoothed = (1.0 - self.smoothing) * list_distribution + self.smoothing * target
            kl = (target[support] * np.log(target[support] / smoothed[:, support])).sum(axis=1)
            objective = (1.0 - self.calibration_weight) * relevance - self.calibration_weight * kl
            objective[picked] = -np.inf
            choice = int(np.argmax(objective))  # ties -> better base rank
            order.append(choice)
            picked[choice] = True
            list_categories += distributions[choice]
        rest = np.flatnonzero(~picked)  # already in base rank order
        return np.concatenate([np.asarray(order, dtype=np.int64), rest])
