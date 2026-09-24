import zipfile
from pathlib import Path

import pandas as pd
import pytest

from recint.data import movielens
from recint.data.movielens import MovieLensVariant, download_movielens, read_movielens_ratings
from recint.data.preprocess import (
    deduplicate_interactions,
    filter_min_rating,
    k_core_filter,
    prepare_interactions,
)


def _frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_filter_min_rating():
    frame = _frame([(1, 1, 5.0, 0), (1, 2, 3.0, 1)])
    assert len(filter_min_rating(frame, None)) == 2
    assert list(filter_min_rating(frame, 4.0)["item_id"]) == [1]


def test_deduplicate_keeps_earliest():
    frame = _frame([(1, 7, 3.0, 5), (1, 7, 4.0, 2), (1, 8, 4.0, 3)])
    result = deduplicate_interactions(frame)
    assert list(result["timestamp"]) == [2, 3]


def test_k_core_is_iterative():
    # Item 3 has one interaction -> dropped; that leaves user 2 with one interaction
    # -> user 2 dropped; that leaves item 2 with one interaction -> dropped.
    frame = _frame(
        [(1, 1, 1.0, 0), (1, 2, 1.0, 1), (3, 1, 1.0, 0), (3, 4, 1.0, 1), (1, 4, 1.0, 2),
         (2, 2, 1.0, 0), (2, 3, 1.0, 1)]
    )
    result = k_core_filter(frame, user_core=2, item_core=2)
    assert set(result["user_id"]) == {1, 3}
    assert set(result["item_id"]) == {1, 4}
    assert (result.groupby("user_id").size() >= 2).all()
    assert (result.groupby("item_id").size() >= 2).all()


def test_prepare_sorts_by_user_then_time_with_stable_ties():
    frame = _frame([(2, 1, 1.0, 5), (1, 2, 1.0, 9), (1, 1, 1.0, 9), (1, 3, 1.0, 1), (2, 3, 1.0, 1)])
    prep = {"min_rating": None, "user_core": 1, "item_core": 1}
    result = prepare_interactions(frame, prep)
    assert list(zip(result["user_id"], result["item_id"])) == [(1, 3), (1, 2), (1, 1), (2, 3), (2, 1)]


@pytest.fixture
def fake_movielens(tmp_path: Path, monkeypatch) -> Path:
    """A tiny ML-1M-format zip registered under a test variant."""
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("fake/ratings.dat", "1::10::5::100\n1::11::3::50\n2::10::4::70\n")
    variant = MovieLensVariant(
        url=source.as_uri(),
        sha256=movielens.sha256_of_file(source),
        ratings_member="fake/ratings.dat",
        separator=b"::",
    )
    monkeypatch.setitem(movielens.MOVIELENS_VARIANTS, "fake", variant)
    return source


def test_download_verifies_and_reads(fake_movielens, tmp_path):
    zip_path = download_movielens("fake", tmp_path / "raw")
    ratings = read_movielens_ratings(zip_path, "fake")
    assert list(ratings.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert ratings.shape == (3, 4)
    assert ratings.loc[1].tolist() == [1, 11, 3.0, 50]


def test_download_rejects_checksum_mismatch(fake_movielens, tmp_path, monkeypatch):
    bad = MovieLensVariant(fake_movielens.as_uri(), "0" * 64, "fake/ratings.dat", b"::")
    monkeypatch.setitem(movielens.MOVIELENS_VARIANTS, "fake", bad)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_movielens("fake", tmp_path / "raw")
    assert not any((tmp_path / "raw").iterdir())  # partial file cleaned up
