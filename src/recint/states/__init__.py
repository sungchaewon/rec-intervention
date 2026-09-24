"""User-state features and grouping.

Implemented: history length. Planned: preference drift/stability,
recommendation uncertainty, popularity exposure.
"""

from typing import Any

from recint.states.base import UserStateExtractor
from recint.states.grouping import GROUPING_METHODS, assign_state_groups
from recint.states.history import HistoryLengthState

STATE_REGISTRY: dict[str, type[UserStateExtractor]] = {
    "history_length": HistoryLengthState,
}


def build_state_extractor(config: dict[str, Any]) -> UserStateExtractor:
    name = config["name"]
    if name not in STATE_REGISTRY:
        raise ValueError(f"Unknown user state {name!r}; available: {sorted(STATE_REGISTRY)}")
    return STATE_REGISTRY[name](**config.get("params", {}))


__all__ = [
    "GROUPING_METHODS",
    "STATE_REGISTRY",
    "HistoryLengthState",
    "UserStateExtractor",
    "assign_state_groups",
    "build_state_extractor",
]
