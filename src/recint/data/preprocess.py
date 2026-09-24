"""Dataset-agnostic interaction filtering and preparation.

Frames use raw-id columns user_id, item_id, timestamp (+ optional rating).
All operations are deterministic; row order is used as the tie-breaker for
equal timestamps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from recint.data.movielens import download_movielens, read_movielens_ratings

USER_ID, ITEM_ID, TIMESTAMP, RATING = "user_id", "item_id", "timestamp", "rating"


def filter_min_rating(frame: pd.DataFrame, min_rating: float | None) -> pd.DataFrame:
    """Keep ratings >= min_rating; None keeps everything (all ratings = implicit feedback)."""
    if min_rating is None:
        return frame
    return frame[frame[RATING] >= min_rating]


def sort_by_user_time(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values([USER_ID, TIMESTAMP], kind="mergesort").reset_index(drop=True)


def deduplicate_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep each user's earliest interaction with an item."""
    ordered = sort_by_user_time(frame)
    return ordered.drop_duplicates([USER_ID, ITEM_ID], keep="first").reset_index(drop=True)


def k_core_filter(frame: pd.DataFrame, user_core: int, item_core: int) -> pd.DataFrame:
    """Iteratively drop users/items with fewer than the given number of interactions."""
    if user_core < 1 or item_core < 1:
        raise ValueError("user_core and item_core must be >= 1")
    while True:
        user_counts = frame[USER_ID].map(frame[USER_ID].value_counts())
        item_counts = frame[ITEM_ID].map(frame[ITEM_ID].value_counts())
        keep = (user_counts >= user_core) & (item_counts >= item_core)
        if keep.all():
            return frame.reset_index(drop=True)
        frame = frame[keep]


def interaction_stats(frame: pd.DataFrame) -> dict[str, float]:
    n_users, n_items = frame[USER_ID].nunique(), frame[ITEM_ID].nunique()
    per_user = frame.groupby(USER_ID).size()
    return {
        "n_interactions": int(len(frame)),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "density": float(len(frame) / (n_users * n_items)) if n_users and n_items else 0.0,
        "history_length_min": int(per_user.min()),
        "history_length_median": float(per_user.median()),
        "history_length_max": int(per_user.max()),
    }


def load_raw_interactions(preparation: dict[str, Any]) -> pd.DataFrame:
    source = preparation["source"]
    if source == "movielens":
        zip_path = download_movielens(
            preparation["variant"], Path(preparation["raw_dir"]), preparation.get("url")
        )
        return read_movielens_ratings(zip_path, preparation["variant"])
    raise ValueError(f"Unknown preparation source {source!r}; expected 'movielens'")


def prepare_interactions(raw: pd.DataFrame, preparation: dict[str, Any]) -> pd.DataFrame:
    """Rating filter -> dedupe -> k-core -> sort by (user, time)."""
    frame = filter_min_rating(raw, preparation.get("min_rating"))
    frame = deduplicate_interactions(frame)
    frame = k_core_filter(frame, preparation["user_core"], preparation["item_core"])
    return sort_by_user_time(frame)
