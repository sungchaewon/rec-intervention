from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recint.analysis.effect import ACTION, USER_ID


@dataclass(frozen=True)
class RoutingData:
    """Router inputs for one period. Column a of the gain matrices is `actions[a]`."""

    user_ids: np.ndarray
    features: np.ndarray  # (n_users, n_features)
    gains: np.ndarray  # (n_users, n_actions), routing metric
    tiebreak_gains: np.ndarray  # (n_users, n_actions), secondary metric
    base: np.ndarray  # (n_users,), routing metric of the base ranking
    actions: list[str]
    feature_names: list[str]

    def selector_labels(self) -> np.ndarray:
        """Best action per user: max routing gain, ties broken by the secondary gain."""
        best = self.gains == self.gains.max(axis=1, keepdims=True)
        return np.where(best, self.tiebreak_gains, -np.inf).argmax(axis=1)


def build_routing_data(
    user_effects: pd.DataFrame, feature_names: Sequence[str], metric: str, tiebreak_metric: str
) -> RoutingData:
    """Reshape a long (user x action) effect table; features come from user states."""
    missing = [f for f in feature_names if f not in user_effects.columns]
    if missing:
        raise ValueError(f"Routing features {missing} are not computed states; add them to `states`")
    actions = list(dict.fromkeys(user_effects[ACTION]))
    users = user_effects[user_effects[ACTION] == actions[0]].set_index(USER_ID)

    def wide(column: str) -> np.ndarray:
        table = user_effects.pivot(index=USER_ID, columns=ACTION, values=column)
        return table.loc[users.index, actions].to_numpy(dtype=np.float64)

    return RoutingData(
        user_ids=users.index.to_numpy(),
        features=users[list(feature_names)].to_numpy(dtype=np.float64),
        gains=wide(f"{metric}_gain"),
        tiebreak_gains=wide(f"{tiebreak_metric}_gain"),
        base=users[f"{metric}_base"].to_numpy(dtype=np.float64),
        actions=actions,
        feature_names=list(feature_names),
    )
