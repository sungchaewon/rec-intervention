"""Beyond-accuracy metrics, score fusion, and harm-aware frontier routing."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from recint.data import InteractionData, ItemCategories
from recint.interventions import ScoreFusion
from recint.metrics.beyond_accuracy import ListQualityMetrics
from recint.models import top_k_items
from recint.routing import NO_OP
from recint.routing.frontier import (
    DamageModel,
    FamilyData,
    evaluate_choices,
    match_threshold,
    strongest_safe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG = PROJECT_ROOT / "configs" / "experiment" / "toy_reranking.yaml"


def _catalog() -> tuple[InteractionData, ItemCategories]:
    # Popularity: item 0 -> 3 users, item 1 -> 2, items 2..4 -> 1 each.
    frame = pd.DataFrame({"user_id": [0, 0, 1, 1, 2, 2, 3, 3], "item_id": [0, 1, 0, 1, 0, 2, 3, 4]})
    data = InteractionData.from_frame(frame)
    categories = pd.DataFrame({"item_id": [0, 1, 2, 3, 4], "category": ["a", "a", "b", "b", "b"]})
    return data, ItemCategories.from_frame(categories, data.item_ids)


def test_list_quality_metrics():
    data, categories = _catalog()
    quality = ListQualityMetrics(data, categories, ["tail", "ild", "calibration"], head_fraction=0.4)
    ranked = np.array([[0, 1], [2, 3]])  # head pair vs tail pair
    values = quality.compute(np.array([0, 0]), ranked, k_values=[2])
    np.testing.assert_allclose(values["tail@2"], [0.0, 1.0])  # head = items 0 and 1
    # Items 0 and 1 are always co-consumed (cosine 1 -> ILD ~0); 2 and 3 never (ILD 1).
    np.testing.assert_allclose(values["ild@2"], [1 - 2 / np.sqrt(6), 1.0], atol=1e-12)
    # User 0's history is all category a: a list of a-items is perfectly calibrated.
    assert values["calibration@2"][0] > values["calibration@2"][1]
    np.testing.assert_allclose(values["calibration@2"][0], 0.0, atol=1e-12)


def test_fusion_weight_zero_keeps_base_order():
    data, _ = _catalog()
    base = np.array([[0.1, 0.9, 0.5, -np.inf, 0.3]])
    fusion = ScoreFusion(weight=0.0, model={"name": "item_cooccurrence"}).fit(data)
    out = fusion.apply(np.array([3]), base)
    np.testing.assert_array_equal(top_k_items(out, 4), top_k_items(base, 4))
    assert np.isneginf(out[0, 3])


def _family(risk: np.ndarray, damage: np.ndarray, objective: np.ndarray) -> FamilyData:
    n_users, n_actions = damage.shape
    strength = np.broadcast_to(np.linspace(0, 1, n_actions)[None, :, None], (n_users, n_actions, 1))
    return FamilyData(
        user_ids=np.arange(n_users),
        actions=[f"f@{i}" for i in range(n_actions)],
        features=np.concatenate([np.repeat(risk[:, None, None], n_actions, axis=1), strength], axis=2),
        objective_gain=objective,
        label_gain=np.where(damage, -1.0, 0.0),
        report_gain=np.where(damage, -1.0, 0.0),
        protect=np.ones(n_users, dtype=bool),
    )


def test_strongest_safe_and_matching():
    p = np.array([[0.1, 0.4, 0.9], [0.8, 0.9, 0.95]])
    np.testing.assert_array_equal(strongest_safe(p, 0.5), [1, NO_OP])
    np.testing.assert_array_equal(strongest_safe(p, 1.0), [2, 2])
    assert match_threshold({0.2: 0.1, 0.5: 0.3, 0.8: 0.5}, target=0.3) == 0.5
    assert match_threshold({0.2: 0.1, 0.5: 0.3}, target=0.9) == 0.5


def test_damage_model_protects_risky_users():
    rng = np.random.default_rng(0)
    risk = rng.uniform(size=400)
    damage = (risk[:, None] + np.linspace(0, 0.3, 3)[None, :]) > 0.8  # risky users and strong actions get damaged
    objective = np.tile(np.linspace(0.1, 0.3, 3), (400, 1))
    data = _family(risk, damage, objective)
    model = DamageModel("logistic", harm_weight=5.0, seed=0).fit(data)
    choices = strongest_safe(model.predict(data), threshold=0.5)
    routed, uniform = evaluate_choices(data, choices), evaluate_choices(data, np.full(400, 2))
    assert routed["n_damaged"] < 0.2 * uniform["n_damaged"]
    assert routed["objective_gain"] > 0.5 * uniform["objective_gain"]


def test_frontier_cli_smoke(tmp_path):
    overrides = ["dataset.params.n_users=200", "dataset.params.n_items=80", "dataset.params.max_history=40",
                 "frontier.seeds=[0]", "frontier.n_thresholds=11"]
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "frontier.py"), "--config", str(TOY_CONFIG),
         "--output-dir", str(tmp_path), *[a for o in overrides for a in ("--override", o)]],
        capture_output=True, text=True, check=True,
    )
    assert "router[gbdt,hw=5]" in completed.stdout
    matched = pd.read_csv(tmp_path / "toy_reranking" / "frontier" / "matched.csv")
    assert set(matched["family"]) == {"popularity", "mmr", "calibration"}
    assert (tmp_path / "toy_reranking" / "frontier" / "curves.csv").exists()
