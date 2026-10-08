"""Policies map each user to an action index, or NO_OP to keep the base ranking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from recint.routing.data import RoutingData

NO_OP = -1


def realized(gains: np.ndarray, choices: np.ndarray) -> np.ndarray:
    """Per-user gain of the chosen action (0 for NO_OP)."""
    picked = np.take_along_axis(gains, np.maximum(choices, 0)[:, None], axis=1).ravel()
    return np.where(choices == NO_OP, 0.0, picked)


def evaluate_policy(data: RoutingData, choices: np.ndarray) -> dict[str, float]:
    """Mean gain on the routing and tiebreak metrics, and Rescue/Damage/Net counts.

    For a binary routing metric such as HR@K, rescued/damaged are Rescue@K/Damage@K.
    """
    gain = realized(data.gains, choices)
    n_rescued, n_damaged = int((gain > 0).sum()), int((gain < 0).sum())
    return {
        "mean_gain": float(gain.mean()),
        "tiebreak_mean_gain": float(realized(data.tiebreak_gains, choices).mean()),
        "n_rescued": n_rescued,
        "n_damaged": n_damaged,
        "net": n_rescued - n_damaged,
        "intervene_rate": float((choices != NO_OP).mean()),
    }


def oracle_choices(data: RoutingData) -> np.ndarray:
    """Hindsight upper bound: the best action per user, NO_OP if none helps."""
    labels = data.selector_labels()
    best_gain = np.take_along_axis(data.gains, labels[:, None], axis=1).ravel()
    return np.where(best_gain > 0, labels, NO_OP)


def best_uniform_action(data: RoutingData) -> int:
    """Action with the highest mean gain on `data`, NO_OP if none is positive."""
    means = data.gains.mean(axis=0)
    return int(means.argmax()) if means.max() > 0 else NO_OP


def best_uniform_choices(fit_data: RoutingData, apply_data: RoutingData) -> np.ndarray:
    return np.full(len(apply_data.user_ids), best_uniform_action(fit_data))


def uncertainty_gate_choices(
    fit_data: RoutingData, apply_data: RoutingData, feature: str, quantiles: Sequence[float]
) -> np.ndarray:
    """Confidence-threshold gate (CoReLLa-style): apply one action to users whose
    `feature` is at or above a quantile; (quantile, action) chosen on `fit_data`."""
    column = fit_data.feature_names.index(feature)
    fit_values, apply_values = fit_data.features[:, column], apply_data.features[:, column]
    best_gain, best_threshold, best_action = 0.0, np.inf, NO_OP
    for quantile in quantiles:
        threshold = float(np.quantile(fit_values, quantile))
        gated = fit_values >= threshold
        totals = fit_data.gains[gated].sum(axis=0) / len(fit_values)
        action = int(totals.argmax())
        if totals[action] > best_gain:
            best_gain, best_threshold, best_action = totals[action], threshold, action
    return np.where(apply_values >= best_threshold, best_action, NO_OP)
