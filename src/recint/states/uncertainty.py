"""Base-model confidence signals computed from base scores (seen items = -inf)."""

from __future__ import annotations

import numpy as np
from scipy.special import entr

from recint.states.base import StateContext, UserStateExtractor


def _top_two(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1], top[:, 0]


class PredictionEntropyState(UserStateExtractor):
    """Normalized entropy of softmax(base_scores / temperature) over unseen items.

    0 = the base model puts all mass on one item (confident), 1 = uniform
    (maximally uncertain). Meaningful as a probability only for models trained
    with a softmax objective (e.g. SASRec with CE); for other scorers it is a
    relative spread measure.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def compute(self, context: StateContext) -> np.ndarray:
        logits = context.base_scores / self.temperature
        finite = np.isfinite(logits)
        row_max = np.where(finite, logits, -np.inf).max(axis=1, keepdims=True)
        weights = np.where(finite, np.exp(logits - np.where(np.isfinite(row_max), row_max, 0.0)), 0.0)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        entropy = entr(probabilities).sum(axis=1)
        n_candidates = finite.sum(axis=1)
        max_entropy = np.log(np.maximum(n_candidates, 1))
        return np.divide(entropy, max_entropy, out=np.zeros_like(entropy), where=n_candidates > 1)


class TopScoreState(UserStateExtractor):
    """Highest base score among unseen items (scale depends on the base model)."""

    def compute(self, context: StateContext) -> np.ndarray:
        return context.base_scores.max(axis=1)


class ScoreMarginState(UserStateExtractor):
    """Gap between the top-1 and top-2 base scores; small = base is torn at the top."""

    def compute(self, context: StateContext) -> np.ndarray:
        first, second = _top_two(context.base_scores)
        return first - second
