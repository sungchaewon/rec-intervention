"""Per-user intervention effects and their summaries.

`user_effects` is long-format: one row per (user, action). For HR@K, a
beneficial/harmful user is exactly Rescue@K/Damage@K (target enters/leaves the
top-K), and net = rescued - damaged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from recint.analysis.bootstrap import bootstrap_mean_ci
from recint.metrics.gain import EffectLabel, classify_effect, intervention_gain

ALL_USERS = "all"
ACTION = "action"
USER_ID = "user_id"


def group_column(state: str) -> str:
    return f"{state}_group"


def build_user_effect_table(
    user_ids: np.ndarray,
    state_values: Mapping[str, np.ndarray],
    state_groups: Mapping[str, np.ndarray],
    base_metrics: Mapping[str, np.ndarray],
    intervention_metrics: Mapping[str, np.ndarray],
    neutral_threshold: float,
    action: str,
    action_features: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Rows for one action: user, state values/groups, per-action features (how the
    action changes the list), and base/intervention/gain/effect per metric."""
    if base_metrics.keys() != intervention_metrics.keys():
        raise ValueError("base and intervention metrics must have the same keys")
    columns: dict[str, np.ndarray | str] = {USER_ID: user_ids, ACTION: action}
    columns.update(state_values)
    columns.update({group_column(name): groups for name, groups in state_groups.items()})
    columns.update(action_features or {})
    for metric, base in base_metrics.items():
        gain = intervention_gain(base, intervention_metrics[metric])
        columns[f"{metric}_base"] = base
        columns[f"{metric}_intervention"] = intervention_metrics[metric]
        columns[f"{metric}_gain"] = gain
        columns[f"{metric}_effect"] = classify_effect(gain, neutral_threshold)
    return pd.DataFrame(columns)


def _wilcoxon_p_value(gain: np.ndarray) -> float:
    """Two-sided paired test of gain != 0; NaN when all gains are zero."""
    if np.count_nonzero(gain) == 0:
        return float("nan")
    return float(stats.wilcoxon(gain, zero_method="wilcox").pvalue)


def _summarize(
    rows: pd.DataFrame, metric: str, n_resamples: int, confidence: float, rng: np.random.Generator
) -> dict[str, float]:
    gain = rows[f"{metric}_gain"].to_numpy()
    effect = rows[f"{metric}_effect"]
    base = rows[f"{metric}_base"].mean()
    ci_low, ci_high = bootstrap_mean_ci(gain, n_resamples, confidence, rng)
    n_rescued = int((effect == EffectLabel.BENEFICIAL).sum())
    n_damaged = int((effect == EffectLabel.HARMFUL).sum())
    return {
        "n_users": len(rows),
        "base": base,
        "intervention": rows[f"{metric}_intervention"].mean(),
        "mean_gain": gain.mean(),
        "gain_ci_low": ci_low,
        "gain_ci_high": ci_high,
        "relative_gain": gain.mean() / base if base > 0 else float("nan"),
        "n_rescued": n_rescued,
        "n_damaged": n_damaged,
        "net": n_rescued - n_damaged,
        "beneficial_rate": n_rescued / len(rows),
        "neutral_rate": (effect == EffectLabel.NEUTRAL).mean(),
        "harm_rate": n_damaged / len(rows),
        "wilcoxon_p": _wilcoxon_p_value(gain),
    }


def user_subsets(
    users: pd.DataFrame, groupings: Mapping[str, Sequence[str]]
) -> list[tuple[str, str, pd.DataFrame]]:
    """(state, group, rows): all users first, then each non-empty state group in label order."""
    subsets = [(ALL_USERS, ALL_USERS, users)]
    for state, labels in groupings.items():
        for label in labels:
            rows = users[users[group_column(state)] == label]
            if not rows.empty:
                subsets.append((state, label, rows))
    return subsets


def summarize_effects(
    user_effects: pd.DataFrame,
    metrics: Sequence[str],
    groupings: Mapping[str, Sequence[str]],
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One row per (action, state, state group, metric); state='all' covers every user.

    Empty groups (possible with tied quantile edges) are omitted.
    """
    rows = []
    for action, action_rows in user_effects.groupby(ACTION, sort=False):
        for state, group, subset in user_subsets(action_rows, groupings):
            for metric in metrics:
                rows.append(
                    {
                        ACTION: action,
                        "state": state,
                        "state_group": group,
                        "metric": metric,
                        **_summarize(subset, metric, n_resamples, confidence, rng),
                    }
                )
    return pd.DataFrame(rows)
