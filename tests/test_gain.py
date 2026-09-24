import numpy as np
import pytest

from recint.metrics import EffectLabel, classify_effect, intervention_gain


def test_gain_is_intervention_minus_base():
    np.testing.assert_allclose(intervention_gain([0.5, 1.0, 0.0], [0.7, 0.2, 0.0]), [0.2, -0.8, 0.0])


def test_gain_shape_mismatch_raises():
    with pytest.raises(ValueError):
        intervention_gain([0.1, 0.2], [0.1])


def test_classify_effect_zero_threshold():
    labels = classify_effect(np.array([0.1, 0.0, -0.1]), neutral_threshold=0.0)
    assert list(labels) == [EffectLabel.BENEFICIAL, EffectLabel.NEUTRAL, EffectLabel.HARMFUL]


def test_classify_effect_threshold_is_inclusive_neutral():
    labels = classify_effect(np.array([0.05, 0.051, -0.05, -0.051]), neutral_threshold=0.05)
    assert list(labels) == ["neutral", "beneficial", "neutral", "harmful"]


def test_negative_threshold_raises():
    with pytest.raises(ValueError):
        classify_effect(np.array([0.0]), neutral_threshold=-0.1)
