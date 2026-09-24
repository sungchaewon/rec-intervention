"""State-conditioned intervention effect tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from recint.analysis.bootstrap import bootstrap_mean_ci
from recint.metrics.gain import EffectLabel, classify_effect, intervention_gain

ALL_USERS_GROUP = "all"


def build_user_effect_table(
    user_ids: np.ndarray,
    state_values: np.ndarray,
    state_groups: np.ndarray,
    base_metrics: dict[str, np.ndarray],
    intervention_metrics: dict[str, np.ndarray],
    neutral_threshold: float,
    intervention_name: str,
) -> pd.DataFrame:
    """One row per evaluated user with base/intervention/gain/effect per metric."""
    if base_metrics.keys() != intervention_metrics.keys():
        raise ValueError("base and intervention metrics must have the same keys")
    columns: dict[str, np.ndarray | str] = {
        "user_id": user_ids,
        "intervention": intervention_name,
        "state_value": state_values,
        "state_group": state_groups,
    }
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


def _summarize_group(
    group: pd.DataFrame,
    metric: str,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    gain = group[f"{metric}_gain"].to_numpy()
    effect = group[f"{metric}_effect"]
    ci_low, ci_high = bootstrap_mean_ci(gain, n_resamples, confidence, rng)
    return {
        "n_users": len(group),
        "state_min": group["state_value"].min(),
        "state_max": group["state_value"].max(),
        "base": group[f"{metric}_base"].mean(),
        "intervention": group[f"{metric}_intervention"].mean(),
        "mean_gain": gain.mean(),
        "gain_ci_low": ci_low,
        "gain_ci_high": ci_high,
        "beneficial_rate": (effect == EffectLabel.BENEFICIAL).mean(),
        "neutral_rate": (effect == EffectLabel.NEUTRAL).mean(),
        "harm_rate": (effect == EffectLabel.HARMFUL).mean(),
        "wilcoxon_p": _wilcoxon_p_value(gain),
    }


def summarize_by_state(
    user_effects: pd.DataFrame,
    metrics: Sequence[str],
    group_order: Sequence[str],
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Per (state group, metric) effect summary, plus an `all` row per metric.

    Empty groups (possible with tied quantile edges) are omitted.
    """
    rows = []
    for metric in metrics:
        groups = [(label, user_effects[user_effects["state_group"] == label]) for label in group_order]
        groups.append((ALL_USERS_GROUP, user_effects))
        for label, group in groups:
            if group.empty:
                continue
            rows.append(
                {
                    "state_group": label,
                    "metric": metric,
                    **_summarize_group(group, metric, n_resamples, confidence, rng),
                }
            )
    return pd.DataFrame(rows)
