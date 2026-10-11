from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recint.analysis.effect import ACTION, USER_ID


@dataclass(frozen=True)
class RoutingData:
    """Router inputs for one period. Column a of every gain matrix is `actions[a]`.

    Three metrics play different roles: `gains` (label metric) trains the router,
    `tiebreak_gains` breaks ties between actions, and `report_gains` (e.g. HR@K,
    giving Rescue/Damage) is what policies are reported on.
    """

    user_ids: np.ndarray
    features: np.ndarray  # (n_users, n_features)
    gains: np.ndarray  # (n_users, n_actions)
    tiebreak_gains: np.ndarray
    report_gains: np.ndarray
    base: np.ndarray  # (n_users,), label metric of the base ranking
    report_base: np.ndarray
    actions: list[str]
    feature_names: list[str]

    def selector_labels(self) -> np.ndarray:
        """Best action per user: max label gain, ties broken by the tiebreak gain."""
        best = self.gains == self.gains.max(axis=1, keepdims=True)
        return np.where(best, self.tiebreak_gains, -np.inf).argmax(axis=1)


def build_routing_data(
    user_effects: pd.DataFrame,
    user_features: Sequence[str],
    action_features: Sequence[str],
    metric: str,
    tiebreak_metric: str,
    report_metric: str,
) -> RoutingData:
    """Reshape a long (user x action) effect table. User features are states;
    action features (how each action changes the list) are spread to one column per action."""
    missing = [f for f in [*user_features, *action_features] if f not in user_effects.columns]
    if missing:
        raise ValueError(f"Routing features {missing} are not in the effect table; add them to `states`")
    actions = list(dict.fromkeys(user_effects[ACTION]))
    users = user_effects[user_effects[ACTION] == actions[0]].set_index(USER_ID)

    def wide(column: str) -> np.ndarray:
        table = user_effects.pivot(index=USER_ID, columns=ACTION, values=column)
        return table.loc[users.index, actions].to_numpy(dtype=np.float64)

    features = [users[list(user_features)].to_numpy(dtype=np.float64)]
    names = list(user_features)
    for column in action_features:
        features.append(wide(column))
        names.extend(f"{column}[{action}]" for action in actions)
    return RoutingData(
        user_ids=users.index.to_numpy(),
        features=np.hstack(features),
        gains=wide(f"{metric}_gain"),
        tiebreak_gains=wide(f"{tiebreak_metric}_gain"),
        report_gains=wide(f"{report_metric}_gain"),
        base=users[f"{metric}_base"].to_numpy(dtype=np.float64),
        report_base=users[f"{report_metric}_base"].to_numpy(dtype=np.float64),
        actions=actions,
        feature_names=names,
    )
