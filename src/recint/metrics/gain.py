"""Per-sample intervention gain and effect classification."""

from __future__ import annotations

from enum import StrEnum

import numpy as np


class EffectLabel(StrEnum):
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    HARMFUL = "harmful"


def intervention_gain(base_metric: np.ndarray, intervention_metric: np.ndarray) -> np.ndarray:
    """gain = metric_after_intervention - metric_before_intervention (per sample)."""
    base_metric = np.asarray(base_metric, dtype=np.float64)
    intervention_metric = np.asarray(intervention_metric, dtype=np.float64)
    if base_metric.shape != intervention_metric.shape:
        raise ValueError("base and intervention metrics must have the same shape")
    return intervention_metric - base_metric


def classify_effect(gain: np.ndarray, neutral_threshold: float) -> np.ndarray:
    """beneficial if gain > t, harmful if gain < -t, otherwise neutral."""
    if neutral_threshold < 0:
        raise ValueError("neutral_threshold must be >= 0")
    gain = np.asarray(gain, dtype=np.float64)
    labels = np.full(gain.shape, EffectLabel.NEUTRAL.value, dtype=object)
    labels[gain > neutral_threshold] = EffectLabel.BENEFICIAL.value
    labels[gain < -neutral_threshold] = EffectLabel.HARMFUL.value
    return labels
