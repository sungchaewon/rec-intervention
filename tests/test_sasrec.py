"""SASRec unit tests and a CPU train -> analyze smoke test on tiny toy data."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from recint.data import InteractionData, leave_last_out
from recint.data.interactions import load_interactions
from recint.models.sasrec import (
    PAD_TOKEN,
    SASRecNetwork,
    SASRecRecommender,
    next_item_training_pairs,
    pad_left,
)
from recint.utils import load_experiment_config, set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SASREC_CONFIG = PROJECT_ROOT / "configs" / "experiment" / "ml-1m_sasrec_popularity_penalty.yaml"
TINY_PARAMS = {
    "max_len": 12, "hidden_dim": 16, "n_blocks": 1, "n_heads": 2, "dropout": 0.1,
    "learning_rate": 0.01, "weight_decay": 0.0, "batch_size": 32, "max_epochs": 3,
    "patience": 2, "eval_k": 10, "eval_batch_size": 64, "device": "cpu",
}  # fmt: skip
TINY_OVERRIDES = [
    "dataset=../dataset/toy.yaml",
    "dataset.params.n_users=120",
    "dataset.params.n_items=60",
    "dataset.params.max_history=30",
    "analysis.bootstrap.n_resamples=20",
    *[f"model.params.{key}={value}" for key, value in TINY_PARAMS.items()],
]


def test_pad_left_keeps_most_recent_items():
    tokens = pad_left([np.array([0, 1, 2, 3]), np.array([5]), np.array([], dtype=np.int64)], max_len=3)
    np.testing.assert_array_equal(tokens, [[2, 3, 4], [0, 0, 6], [0, 0, 0]])


def test_training_pairs_shift_by_one_and_skip_short_users():
    inputs, targets = next_item_training_pairs([np.array([0, 1, 2]), np.array([7])], max_len=3)
    np.testing.assert_array_equal(inputs, [[0, 1, 2]])
    np.testing.assert_array_equal(targets, [[0, 2, 3]])


def test_user_sequences_follow_time_order():
    frame = pd.DataFrame({"user_id": [1, 1, 1, 2], "item_id": [10, 11, 12, 10], "timestamp": [3, 1, 2, 0]})
    data = InteractionData.from_frame(frame, timestamp_col="timestamp")
    sequences = data.user_sequences()
    assert [data.item_ids[s].tolist() for s in sequences] == [[11, 12, 10], [10]]


def test_network_is_causal():
    torch.manual_seed(0)
    network = SASRecNetwork(n_items=20, max_len=6, hidden_dim=8, n_blocks=2, n_heads=2, dropout=0.0).eval()
    tokens = torch.tensor([[PAD_TOKEN, 3, 5, 7, 9, 11]])
    changed_future = tokens.clone()
    changed_future[0, 4:] = torch.tensor([2, 4])
    with torch.no_grad():
        original, altered = network(tokens), network(changed_future)
    torch.testing.assert_close(original[:, :4], altered[:, :4])
    assert not torch.allclose(original[:, 4:], altered[:, 4:])
    assert torch.isfinite(original).all()


@pytest.fixture(scope="module")
def toy_train() -> InteractionData:
    config = load_experiment_config(SASREC_CONFIG, TINY_OVERRIDES)
    data = load_interactions(config["dataset"], np.random.default_rng(0))
    return leave_last_out(data, min_train_interactions=2).train


@pytest.fixture(scope="module")
def fitted(toy_train) -> SASRecRecommender:
    set_global_seed(0)
    return SASRecRecommender(**TINY_PARAMS).fit(toy_train)


def test_fit_records_history_and_scores_catalog(fitted, toy_train):
    assert 1 <= len(fitted.history) <= TINY_PARAMS["max_epochs"]
    scores = fitted.score(np.arange(5))
    assert scores.shape == (5, toy_train.n_items)
    assert np.isfinite(scores).all()


def test_fit_is_deterministic_on_cpu(fitted, toy_train):
    set_global_seed(0)
    again = SASRecRecommender(**TINY_PARAMS).fit(toy_train)
    np.testing.assert_allclose(again.score(np.arange(5)), fitted.score(np.arange(5)))


def test_save_load_roundtrip(fitted, toy_train, tmp_path):
    path = tmp_path / "model.pt"
    fitted.save(path)
    loaded = SASRecRecommender.load(path, toy_train, **TINY_PARAMS)
    np.testing.assert_allclose(loaded.score(np.arange(5)), fitted.score(np.arange(5)))


def test_load_rejects_other_data_and_architecture(fitted, toy_train, tmp_path):
    path = tmp_path / "model.pt"
    fitted.save(path)
    other = toy_train.with_interactions(toy_train.interactions.iloc[1:])
    with pytest.raises(ValueError, match="different data"):
        SASRecRecommender.load(path, other, **TINY_PARAMS)
    with pytest.raises(ValueError, match="architecture"):
        SASRecRecommender.load(path, toy_train, **{**TINY_PARAMS, "hidden_dim": 32})


def test_train_then_analyze_cli(tmp_path):
    checkpoint = tmp_path / "ckpt" / "model.pt"
    overrides = [*TINY_OVERRIDES, f"model_checkpoint={checkpoint.as_posix()}"]
    override_args = [arg for override in overrides for arg in ("--override", override)]
    scripts = PROJECT_ROOT / "scripts"
    subprocess.run(
        [sys.executable, str(scripts / "train.py"), "--config", str(SASREC_CONFIG), *override_args],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert checkpoint.exists()
    assert (checkpoint.parent / "train_history.csv").exists()
    analyzed = subprocess.run(
        [sys.executable, str(scripts / "analyze_intervention.py"), "--config", str(SASREC_CONFIG),
         "--output-dir", str(tmp_path / "out"), *override_args],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert "harm_rate" in analyzed.stdout
