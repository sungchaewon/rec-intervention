import numpy as np
import pandas as pd

from recint.analysis import (
    ALL_USERS,
    NO_OP_ACTION,
    bootstrap_mean_ci,
    build_user_effect_table,
    routing_headroom,
    summarize_effects,
)
from recint.data import InteractionData
from recint.interventions import IdentityIntervention, PopularityPenaltyReranker
from recint.models import top_k_items


def _train_data() -> InteractionData:
    # Item popularity: item 0 -> 3, item 1 -> 2, item 2 -> 1, item 3 -> 0.
    frame = pd.DataFrame({"user_id": [0, 1, 2, 0, 1, 0, 3], "item_id": [0, 0, 0, 1, 1, 2, 3]})
    data = InteractionData.from_frame(frame)
    # Drop item 3's interaction but keep it in the catalog (popularity 0).
    return data.with_interactions(data.interactions[data.interactions["item_idx"] != 3])


def test_zero_penalty_preserves_candidate_order():
    base = np.array([[0.9, 0.1, 0.5, 0.3]])
    reranker = PopularityPenaltyReranker(penalty_weight=0.0, candidate_size=3).fit(_train_data())
    intervened = reranker.apply(np.array([0]), base)
    np.testing.assert_array_equal(top_k_items(intervened, 3), top_k_items(base, 3))
    assert np.isneginf(intervened[0, 1])  # outside candidates


def test_penalty_demotes_popular_items_and_keeps_masked_items():
    base = np.array([[0.9, 0.8, -np.inf, 0.1]])
    reranker = PopularityPenaltyReranker(penalty_weight=2.0, candidate_size=4).fit(_train_data())
    intervened = reranker.apply(np.array([0]), base)
    assert top_k_items(intervened, 1)[0, 0] == 3  # unpopular item promoted
    assert np.isneginf(intervened[0, 2])
    assert np.isfinite(base[0, 0])  # base scores untouched


def test_identity_intervention_copies():
    base = np.array([[0.2, 0.4]])
    out = IdentityIntervention().apply(np.array([0]), base)
    np.testing.assert_array_equal(out, base)
    assert out is not base


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    values = np.random.default_rng(0).normal(size=200)
    low, high = bootstrap_mean_ci(values, 500, 0.95, np.random.default_rng(1))
    assert low < values.mean() < high
    assert (low, high) == bootstrap_mean_ci(values, 500, 0.95, np.random.default_rng(1))


def test_state_summary_rates_and_groups():
    table = _effects("test", base=[0.5, 0.5, 0.2, 0.2], intervened=[0.7, 0.3, 0.2, 0.6])
    summary = summarize_effects(
        table, ["ndcg@10"], {"history": ["short", "medium", "long"]}, 100, 0.95, np.random.default_rng(0)
    ).set_index("state_group")

    assert list(summary.index) == [ALL_USERS, "short", "long"]  # empty "medium" omitted
    assert summary.loc["short", "harm_rate"] == 0.5
    assert summary.loc["long", "neutral_rate"] == 0.5
    np.testing.assert_allclose(summary.loc[ALL_USERS, "mean_gain"], 0.1)
    assert summary.loc[ALL_USERS, ["n_rescued", "n_damaged", "net"]].tolist() == [2, 1, 1]
    rates = summary[["beneficial_rate", "neutral_rate", "harm_rate"]].sum(axis=1)
    np.testing.assert_allclose(rates, 1.0)


def test_routing_headroom_matches_hand_computation():
    base = [0.5, 0.5, 0.2, 0.4]
    table = pd.concat(
        [
            _effects("a", base, intervened=np.add(base, [0.2, -0.2, 0.0, 0.4])),
            _effects("b", base, intervened=np.add(base, [-0.1, 0.1, 0.3, -0.4])),
        ]
    )
    row = routing_headroom(table, ["ndcg@10"], {}).iloc[0]
    assert row["best_uniform_action"] == "a"
    np.testing.assert_allclose(row["best_uniform_gain"], 0.1)  # mean of a's gains
    np.testing.assert_allclose(row["oracle_gain"], 0.25)  # mean of [0.2, 0.1, 0.3, 0.4]
    np.testing.assert_allclose(row["oracle_gap"], 0.15)
    assert row["oracle_intervene_rate"] == 1.0


def test_headroom_prefers_no_op_when_every_action_hurts():
    base = [0.5, 0.5]
    table = _effects("a", base, intervened=[0.4, 0.55])  # mean gain < 0, one user rescued
    table = pd.concat([table, _effects("b", base, intervened=[0.3, 0.3])])
    row = routing_headroom(table, ["ndcg@10"], {}).iloc[0]
    assert row["best_uniform_action"] == NO_OP_ACTION
    assert row["best_uniform_gain"] == 0.0
    np.testing.assert_allclose(row["oracle_gain"], 0.025)
    assert row["oracle_intervene_rate"] == 0.5


def _effects(action: str, base, intervened) -> pd.DataFrame:
    n_users = len(base)
    groups = np.array(["short", "short", "long", "long"][:n_users], dtype=object)
    return build_user_effect_table(
        user_ids=np.arange(n_users),
        state_values={"history": np.arange(n_users, dtype=float)},
        state_groups={"history": groups},
        base_metrics={"ndcg@10": np.asarray(base, dtype=float)},
        intervention_metrics={"ndcg@10": np.asarray(intervened, dtype=float)},
        neutral_threshold=0.0,
        action=action,
    )
