"""Routing headroom: how much a per-user choice of action could beat any uniform policy.

The no-op (keep the base ranking) is always an available action with gain 0,
so the best uniform policy is never worse than the base, and the oracle picks
max(0, best action gain) per user.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from recint.analysis.effect import ACTION, ALL_USERS, USER_ID, user_subsets

NO_OP_ACTION = "none"


def routing_headroom(
    user_effects: pd.DataFrame, metrics: Sequence[str], groupings: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    """One row per (state, state group, metric) with best-uniform vs oracle gain."""
    first_action = user_effects[ACTION].iloc[0]
    users = user_effects[user_effects[ACTION] == first_action].set_index(USER_ID)
    rows = []
    for metric in metrics:
        gains = user_effects.pivot(index=USER_ID, columns=ACTION, values=f"{metric}_gain")
        for state, group, subset in user_subsets(users.reset_index(), groupings):
            subset_gains = gains.loc[subset[USER_ID]]
            uniform = subset_gains.mean()
            best_action = uniform.idxmax()
            best_uniform_gain = uniform.max()
            if best_uniform_gain <= 0:
                best_action, best_uniform_gain = NO_OP_ACTION, 0.0
            best_per_user = subset_gains.max(axis=1).clip(lower=0.0)
            oracle_gain = best_per_user.mean()
            rows.append(
                {
                    "state": state,
                    "state_group": group,
                    "metric": metric,
                    "n_users": len(subset),
                    "base": subset[f"{metric}_base"].mean(),
                    "best_uniform_action": best_action,
                    "best_uniform_gain": best_uniform_gain,
                    "oracle_gain": oracle_gain,
                    "oracle_gap": oracle_gain - best_uniform_gain,
                    "oracle_intervene_rate": float((best_per_user > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["ALL_USERS", "NO_OP_ACTION", "routing_headroom"]
