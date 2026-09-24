import numpy as np
import pytest

from recint.metrics import compute_ranking_metrics, hit_rate_at_k, ndcg_at_k

RANKED = np.array(
    [
        [5, 3, 7, 1],  # target 5 at rank 1
        [2, 9, 4, 8],  # target 4 at rank 3
        [0, 1, 2, 3],  # target 6 missing
    ]
)
TARGETS = np.array([5, 4, 6])


def test_hit_rate_single_target():
    np.testing.assert_array_equal(hit_rate_at_k(RANKED, TARGETS, k=4), [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(hit_rate_at_k(RANKED, TARGETS, k=2), [1.0, 0.0, 0.0])


def test_ndcg_single_target():
    expected = [1.0, 1.0 / np.log2(4), 0.0]
    np.testing.assert_allclose(ndcg_at_k(RANKED, TARGETS, k=4), expected)
    np.testing.assert_allclose(ndcg_at_k(RANKED, TARGETS, k=2), [1.0, 0.0, 0.0])


def test_ndcg_multiple_targets():
    ranked = np.array([[1, 2, 3, 4]])
    targets = [np.array([2, 4])]  # hits at ranks 2 and 4
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    ideal = 1 / np.log2(2) + 1 / np.log2(3)
    np.testing.assert_allclose(ndcg_at_k(ranked, targets, k=4), [dcg / ideal])
    np.testing.assert_array_equal(hit_rate_at_k(ranked, targets, k=1), [0.0])


def test_ndcg_ideal_is_truncated_at_k():
    ranked = np.array([[1, 2]])
    targets = [np.array([1, 2, 3])]  # more relevant items than k
    np.testing.assert_allclose(ndcg_at_k(ranked, targets, k=2), [1.0])


def test_compute_ranking_metrics_names():
    result = compute_ranking_metrics(RANKED, TARGETS, ["hr", "ndcg"], [2, 4])
    assert set(result) == {"hr@2", "hr@4", "ndcg@2", "ndcg@4"}


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        hit_rate_at_k(RANKED, TARGETS, k=5)
