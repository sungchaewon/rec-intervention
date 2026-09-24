import numpy as np
import pandas as pd

from recint.analysis import ALL_USERS_GROUP, bootstrap_mean_ci, build_user_effect_table, summarize_by_state
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
    table = build_user_effect_table(
        user_ids=np.arange(4),
        state_values=np.array([1.0, 2.0, 10.0, 11.0]),
        state_groups=np.array(["short", "short", "long", "long"], dtype=object),
        base_metrics={"ndcg@10": np.array([0.5, 0.5, 0.2, 0.2])},
        intervention_metrics={"ndcg@10": np.array([0.7, 0.3, 0.2, 0.6])},
        neutral_threshold=0.0,
        intervention_name="test",
    )
    summary = summarize_by_state(
        table, ["ndcg@10"], ["short", "medium", "long"], 100, 0.95, np.random.default_rng(0)
    ).set_index("state_group")

    assert list(summary.index) == ["short", "long", ALL_USERS_GROUP]  # empty "medium" omitted
    assert summary.loc["short", "harm_rate"] == 0.5
    assert summary.loc["long", "neutral_rate"] == 0.5
    np.testing.assert_allclose(summary.loc[ALL_USERS_GROUP, "mean_gain"], 0.1)
    rates = summary[["beneficial_rate", "neutral_rate", "harm_rate"]].sum(axis=1)
    np.testing.assert_allclose(rates, 1.0)
