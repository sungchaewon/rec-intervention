"""End-to-end smoke tests on a shrunken toy config."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from recint.analysis import ALL_USERS, NO_OP_ACTION
from recint.experiment import run_intervention_experiment
from recint.utils import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG = PROJECT_ROOT / "configs" / "experiment" / "toy_reranking.yaml"
SMALL_OVERRIDES = [
    "dataset.params.n_users=150",
    "dataset.params.n_items=80",
    "dataset.params.max_history=40",
    "analysis.bootstrap.n_resamples=50",
]


def _run(overrides: list[str]):
    return run_intervention_experiment(load_experiment_config(TOY_CONFIG, SMALL_OVERRIDES + overrides))


def test_toy_pipeline_outputs():
    result = _run([])
    actions = {"popularity", "mmr", "calibration"}
    n_users = result.user_effects["user_id"].nunique()
    assert set(result.user_effects["action"]) == actions
    assert len(result.user_effects) == n_users * len(actions)
    for state in ("history_length", "prediction_entropy", "top_score", "score_margin"):
        assert result.user_effects[state].notna().all()

    summary = result.summary
    assert set(summary["action"]) == actions
    assert set(summary["metric"]) == {"hr@10", "ndcg@10"}
    assert set(summary.loc[summary["state"] == "history_length", "state_group"]) == {"short", "medium", "long"}
    overall = summary[summary["state"] == ALL_USERS]
    assert (overall["n_users"] == n_users).all()
    assert (overall["net"] == overall["n_rescued"] - overall["n_damaged"]).all()


def test_headroom_oracle_bounds_best_uniform():
    headroom = _run([]).headroom
    assert (headroom["oracle_gain"] >= headroom["best_uniform_gain"] - 1e-12).all()
    assert (headroom["best_uniform_gain"] >= 0).all()


def test_pipeline_is_deterministic():
    first, second = _run([]), _run([])
    assert first.summary.equals(second.summary)
    assert first.headroom.equals(second.headroom)


def test_identity_intervention_has_zero_gain_and_no_headroom():
    result = _run(["interventions=[{name: identity}]"])
    assert np.all(result.user_effects["ndcg@10_gain"] == 0.0)
    assert np.all(result.user_effects["ndcg@10_effect"] == "neutral")
    assert (result.headroom["best_uniform_action"] == NO_OP_ACTION).all()
    assert (result.headroom["oracle_gap"] == 0.0).all()


def test_list_override_and_validation():
    config = load_experiment_config(TOY_CONFIG, ["interventions.0.params.penalty_weight=0.2"])
    assert config["interventions"][0]["params"]["penalty_weight"] == 0.2
    with pytest.raises(ValueError, match="candidate_size"):
        _run(["interventions.0.params.candidate_size=5"])
    with pytest.raises(ValueError, match="unique"):
        _run(["interventions=[{name: identity}, {name: identity}]"])


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
    assert "Routing headroom" in completed.stdout
    run_dir = tmp_path / "toy_reranking"
    for name in ("config.yaml", "user_effects.csv", "summary.csv", "headroom.csv"):
        assert (run_dir / name).exists()
