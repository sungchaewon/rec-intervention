"""Routing: validation-period labels, policies, routers, and the route.py CLI."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from recint.data import InteractionData, leave_last_out
from recint.experiment import action_label, expand_interventions, list_change_features, validation_split
from recint.models import ItemCooccurrenceRecommender
from recint.routing import (
    NO_OP,
    GateSelectorRouter,
    RoutingData,
    SingleStageRouter,
    best_uniform_choices,
    build_routing_data,
    evaluate_policy,
    oracle_choices,
    uncertainty_gate_choices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG = PROJECT_ROOT / "configs" / "experiment" / "toy_reranking.yaml"


def test_expand_interventions_grid():
    expanded = expand_interventions(
        [{"name": "mmr", "params": {"candidate_size": 50}, "grid": {"relevance_weight": [0.9, 0.6]}},
         {"name": "identity"}]
    )
    assert [action_label(c) for c in expanded] == ["mmr@0.9", "mmr@0.6", "identity"]
    assert expanded[1]["params"] == {"candidate_size": 50, "relevance_weight": 0.6}


def test_validation_split_hides_test_targets():
    frame = pd.DataFrame({"user_id": [0] * 4 + [1] * 3, "item_id": [0, 1, 2, 3, 1, 2, 4],
                          "timestamp": [0, 1, 2, 3, 0, 1, 2]})
    data = InteractionData.from_frame(frame, timestamp_col="timestamp")
    split = leave_last_out(data, min_train_interactions=2)
    validation = validation_split(split)
    np.testing.assert_array_equal(validation.targets, [2, 2])  # second-to-last items
    history = validation.train.user_sequences()
    assert [h.tolist() for h in history] == [[0, 1], [1]]
    assert not set(split.targets) & set(np.concatenate(history))


def test_score_with_history_uses_given_interactions():
    frame = pd.DataFrame({"user_id": [0, 0, 1, 1], "item_id": [0, 1, 1, 2]})
    data = InteractionData.from_frame(frame)
    model = ItemCooccurrenceRecommender().fit(data)
    shortened = data.with_interactions(data.interactions.iloc[[0, 2]])
    full, short = model.score(np.array([0, 1])), model.score(np.array([0, 1]), history=shortened)
    assert not np.allclose(full, short)
    # Scores come from the given history with the fitted similarities: user 0 now has item 0 only.
    np.testing.assert_allclose(short[0], model.score(np.array([0]), history=data.with_interactions(
        pd.DataFrame({"user_idx": [0], "item_idx": [0]})))[0])


def _routing_data(features, gains, base, tiebreak=None) -> RoutingData:
    gains = np.asarray(gains, dtype=float)
    return RoutingData(
        user_ids=np.arange(len(gains)),
        features=np.asarray(features, dtype=float).reshape(len(gains), -1),
        gains=gains,
        tiebreak_gains=gains if tiebreak is None else np.asarray(tiebreak, dtype=float),
        report_gains=gains,
        base=np.asarray(base, dtype=float),
        report_base=np.asarray(base, dtype=float),
        actions=[f"a{i}" for i in range(gains.shape[1])],
        feature_names=[f"f{i}" for i in range(np.asarray(features).reshape(len(gains), -1).shape[1])],
    )


def test_policy_evaluation_and_baselines():
    data = _routing_data(
        features=[[0.1], [0.2], [0.9], [0.95]],
        gains=[[0, -1], [-1, 0], [1, 0], [0, 1]],
        base=[1, 1, 0, 0],
    )
    np.testing.assert_array_equal(oracle_choices(data), [NO_OP, NO_OP, 0, 1])
    result = evaluate_policy(data, np.array([0, 0, 0, NO_OP]))
    assert (result["n_rescued"], result["n_damaged"], result["net"]) == (1, 1, 0)
    assert result["intervene_rate"] == 0.75
    assert (best_uniform_choices(data, data) == NO_OP).all()  # both actions average 0
    gate = uncertainty_gate_choices(data, data, "f0", quantiles=[0.5])
    np.testing.assert_array_equal(gate, [NO_OP, NO_OP, 0, 0])


def _learnable_data(n_users: int, seed: int) -> RoutingData:
    """Action 1 rescues users with feature > 0.8; everything else harms base-correct users."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n_users, 1))
    rescuable = x[:, 0] > 0.8
    gains = np.zeros((n_users, 2))
    gains[rescuable, 1] = 1.0
    gains[~rescuable] = -1.0
    return _routing_data(x, gains, base=(~rescuable).astype(float))


def test_gate_selector_learns_when_and_what():
    train, test = _learnable_data(600, seed=0), _learnable_data(400, seed=1)
    router = GateSelectorRouter(harm_weight=5.0, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
                                classifier="logistic", n_folds=3, seed=0).fit(train)
    choices = router.choose(test)
    result = evaluate_policy(test, choices)
    oracle = evaluate_policy(test, oracle_choices(test))
    assert result["n_damaged"] <= 0.05 * len(test.user_ids)
    assert result["report_gain"] > 0.8 * oracle["report_gain"]
    assert set(np.unique(choices)) <= {NO_OP, 1}
    single = evaluate_policy(test, SingleStageRouter("logistic", seed=0).fit(train).choose(test))
    assert single["report_gain"] > 0


def test_route_cli_smoke(tmp_path):
    overrides = ["dataset.params.n_users=200", "dataset.params.n_items=80", "dataset.params.max_history=40"]
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "route.py"), "--config", str(TOY_CONFIG),
         "--output-dir", str(tmp_path), *[a for o in overrides for a in ("--override", o)]],
        capture_output=True, text=True, check=True,
    )
    assert "gate_selector[gbdt]" in completed.stdout and "oracle" in completed.stdout
    policies = pd.read_csv(tmp_path / "toy_reranking" / "routing" / "policies.csv")
    oracle = policies.set_index("policy").loc["oracle", "report_gain"]
    assert (policies["report_gain"] <= oracle + 1e-12).all()


def test_route_cli_reuses_cached_effects(tmp_path):
    overrides = ["dataset.params.n_users=200", "dataset.params.n_items=80", "dataset.params.max_history=40"]
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "route.py"), "--config", str(TOY_CONFIG),
               "--output-dir", str(tmp_path), *[a for o in overrides for a in ("--override", o)]]
    subprocess.run(command, capture_output=True, text=True, check=True)
    # Changing only routing settings reuses effects; changing an action recomputes them.
    reused = subprocess.run([*command, "--override", "routing.harm_weight=2"], capture_output=True, text=True, check=True)
    assert "reusing cached per-user effects" in reused.stderr
    changed = subprocess.run([*command, "--override", "interventions.1.grid.relevance_weight=[0.7]"],
                             capture_output=True, text=True, check=True)
    assert "reusing cached" not in changed.stderr
    assert "test period" in changed.stderr


def test_list_change_features():
    base_ranking = np.array([[5, 6, 7, 8, 9]])
    intervened_top = np.array([[5, 8, 9]])  # 8 (base rank 4) and 9 (rank 5) enter the top-3
    features = list_change_features(base_ranking, intervened_top)
    np.testing.assert_array_equal(features["changed@3"], [2.0])
    np.testing.assert_allclose(features["promoted_rank@3"], [4.5])
    unchanged = list_change_features(base_ranking, np.array([[6, 5, 7]]))
    assert unchanged["changed@3"][0] == 0 and unchanged["promoted_rank@3"][0] == 0


def test_build_routing_data_spreads_action_features():
    rows = []
    for action, gain in [("a", [0.5, -0.5]), ("b", [0.0, 1.0])]:
        for user in range(2):
            rows.append({"user_id": user, "action": action, "s": float(user), "changed@10": 1.0 + user,
                         "ndcg@10_gain": gain[user], "ndcg@10_base": 0.2, "hr@10_gain": gain[user], "hr@10_base": 1.0})
    data = build_routing_data(pd.DataFrame(rows), ["s"], ["changed@10"], "ndcg@10", "hr@10", "hr@10")
    assert data.feature_names == ["s", "changed@10[a]", "changed@10[b]"]
    np.testing.assert_array_equal(data.features, [[0, 1, 1], [1, 2, 2]])
    np.testing.assert_array_equal(data.gains, [[0.5, 0.0], [-0.5, 1.0]])
