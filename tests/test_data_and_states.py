import numpy as np
import pandas as pd
import pytest

from recint.data import InteractionData, leave_last_out
from recint.states import (
    HistoryLengthState,
    PredictionEntropyState,
    RankGapState,
    ScoreMarginState,
    StateContext,
    TopEntropyState,
    TopScoreState,
    assign_state_groups,
)


@pytest.fixture
def interactions() -> InteractionData:
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
            "item_id": ["a", "b", "c", "a", "c", "b"],
            "timestamp": [3, 1, 2, 5, 4, 1],
        }
    )
    return InteractionData.from_frame(frame, timestamp_col="timestamp")


def test_from_frame_indexing(interactions):
    assert interactions.n_users == 3
    assert interactions.n_items == 3
    assert list(interactions.user_ids) == ["u1", "u2", "u3"]


def test_leave_last_out_uses_timestamp(interactions):
    split = leave_last_out(interactions, min_train_interactions=1)
    item_index = {item: i for i, item in enumerate(interactions.item_ids)}
    # u1's last item by time is "a" (t=3); u2's is "a" (t=5); u3 has too few interactions.
    np.testing.assert_array_equal(split.eval_users, [0, 1])
    np.testing.assert_array_equal(split.targets, [item_index["a"], item_index["a"]])
    np.testing.assert_array_equal(split.train.history_lengths(), [2, 1, 1])


def test_history_length_state(interactions):
    context = StateContext(interactions, eval_users=np.array([2, 0]), base_scores=np.zeros((2, 3)))
    np.testing.assert_array_equal(HistoryLengthState().compute(context), [1.0, 3.0])


def test_confidence_states():
    scores = np.array([[2.0, 1.0, -np.inf], [0.0, 0.0, 0.0]])
    context = StateContext(train=None, eval_users=np.arange(2), base_scores=scores)  # type: ignore[arg-type]
    entropy = PredictionEntropyState().compute(context)
    assert 0.0 < entropy[0] < 1.0  # two candidates, unequal mass
    np.testing.assert_allclose(entropy[1], 1.0)  # uniform over three candidates
    np.testing.assert_allclose(TopScoreState().compute(context), [2.0, 0.0])
    np.testing.assert_allclose(ScoreMarginState().compute(context), [1.0, 0.0])


def test_boundary_states():
    scores = np.array([[5.0, 4.0, 1.0, 0.0], [1.0, 1.0, 1.0, -np.inf]])
    context = StateContext(train=None, eval_users=np.arange(2), base_scores=scores)  # type: ignore[arg-type]
    np.testing.assert_allclose(RankGapState(upper_rank=1, lower_rank=2).compute(context), [1.0, 0.0])
    np.testing.assert_allclose(RankGapState(upper_rank=2, lower_rank=4).compute(context), [4.0, 0.0])  # -inf -> 0
    entropy = TopEntropyState(top_n=3).compute(context)
    assert entropy[0] < entropy[1]
    np.testing.assert_allclose(entropy[1], 1.0)


def test_fixed_grouping_edges_go_to_lower_group():
    groups = assign_state_groups(
        np.array([1, 5, 6, 20, 21]), "fixed", cut_points=[5, 20], labels=["short", "medium", "long"]
    )
    assert list(groups) == ["short", "short", "medium", "medium", "long"]


def test_quantile_grouping_balances_distinct_values():
    groups = assign_state_groups(
        np.arange(1, 10), "quantile", cut_points=[1 / 3, 2 / 3], labels=["short", "medium", "long"]
    )
    assert list(groups).count("short") == 3
    assert list(groups).count("long") == 3


def test_grouping_label_count_mismatch_raises():
    with pytest.raises(ValueError):
        assign_state_groups(np.arange(5), "fixed", cut_points=[2], labels=["a", "b", "c"])
