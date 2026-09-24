"""End-to-end smoke tests on a shrunken toy config."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from recint.experiment import run_intervention_experiment
from recint.utils import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG = PROJECT_ROOT / "configs" / "experiment" / "toy_popularity_penalty.yaml"
SMALL_OVERRIDES = [
    "dataset.params.n_users=150",
    "dataset.params.n_items=80",
    "dataset.params.max_history=40",
    "analysis.bootstrap.n_resamples=50",
]


def _run(overrides: list[str]):
    return run_intervention_experiment(load_experiment_config(TOY_CONFIG, SMALL_OVERRIDES + overrides))


def test_toy_pipeline_produces_state_summary():
    result = _run([])
    summary = result.state_summary
    assert set(summary["state_group"]) == {"short", "medium", "long", "all"}
    assert set(summary["metric"]) == {"hr@10", "ndcg@10"}
    all_rows = summary[summary["state_group"] == "all"]
    assert (all_rows["n_users"] == len(result.user_effects)).all()


def test_pipeline_is_deterministic():
    first, second = _run([]), _run([])
    assert first.state_summary.equals(second.state_summary)


def test_identity_intervention_has_zero_gain():
    result = _run(["intervention={name: identity, params: {}}"])
    assert np.all(result.user_effects["ndcg@10_gain"] == 0.0)
    assert np.all(result.user_effects["ndcg@10_effect"] == "neutral")


def test_candidate_size_smaller_than_k_raises():
    with pytest.raises(ValueError, match="candidate_size"):
        _run(["intervention.params.candidate_size=5"])


def test_cli_smoke(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analyze_intervention.py"),
            "--config",
            str(TOY_CONFIG),
            "--output-dir",
            str(tmp_path),
            *[arg for override in SMALL_OVERRIDES for arg in ("--override", override)],
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "harm_rate" in completed.stdout
    run_dir = tmp_path / "toy_popularity_penalty"
    for name in ("config.yaml", "user_effects.csv", "state_summary.csv"):
        assert (run_dir / name).exists()
