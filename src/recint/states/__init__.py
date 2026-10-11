"""User-state features and grouping.

Implemented: history length, popularity, category entropy; base-model
confidence (prediction entropy, top score, score margin); top-K boundary
(rank gap, top-N entropy). Planned: preference drift, rating variance,
cold-start exposure.
"""

from typing import Any

from recint.states.base import StateContext, UserStateExtractor
from recint.states.grouping import GROUPING_METHODS, assign_state_groups
from recint.states.history import HistoryCategoryEntropyState, HistoryLengthState, HistoryPopularityState
from recint.states.uncertainty import (
    PredictionEntropyState,
    RankGapState,
    ScoreMarginState,
    TopEntropyState,
    TopScoreState,
)

STATE_REGISTRY: dict[str, type[UserStateExtractor]] = {
    "history_length": HistoryLengthState,
    "history_popularity": HistoryPopularityState,
    "history_category_entropy": HistoryCategoryEntropyState,
    "prediction_entropy": PredictionEntropyState,
    "top_score": TopScoreState,
    "score_margin": ScoreMarginState,
    "rank_gap": RankGapState,
    "top_entropy": TopEntropyState,
}


def build_state_extractor(config: dict[str, Any]) -> UserStateExtractor:
    name = config["name"]
    if name not in STATE_REGISTRY:
        raise ValueError(f"Unknown user state {name!r}; available: {sorted(STATE_REGISTRY)}")
    return STATE_REGISTRY[name](**config.get("params", {}))


__all__ = [
    "GROUPING_METHODS",
    "STATE_REGISTRY",
    "HistoryCategoryEntropyState",
    "HistoryLengthState",
    "HistoryPopularityState",
    "PredictionEntropyState",
    "RankGapState",
    "ScoreMarginState",
    "StateContext",
    "TopEntropyState",
    "TopScoreState",
    "UserStateExtractor",
    "assign_state_groups",
    "build_state_extractor",
]
