"""Implicit-feedback interaction data: (user, item, optional timestamp)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recint.data.synthetic import generate_synthetic_interactions

USER_ID = "user_id"
ITEM_ID = "item_id"
TIMESTAMP = "timestamp"
USER_IDX = "user_idx"
ITEM_IDX = "item_idx"
_ORDER = "_order"


@dataclass(frozen=True)
class InteractionData:
    """Interactions with contiguous integer indices.

    `interactions` has columns `user_idx`, `item_idx` and optionally `timestamp`.
    `user_ids[i]` / `item_ids[j]` map indices back to raw ids. The index space is
    shared between a full dataset and its train split.
    """

    interactions: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        user_col: str = USER_ID,
        item_col: str = ITEM_ID,
        timestamp_col: str | None = None,
    ) -> InteractionData:
        user_codes, user_ids = pd.factorize(frame[user_col], sort=True)
        item_codes, item_ids = pd.factorize(frame[item_col], sort=True)
        interactions = pd.DataFrame({USER_IDX: user_codes, ITEM_IDX: item_codes})
        if timestamp_col is not None:
            interactions[TIMESTAMP] = frame[timestamp_col].to_numpy()
        return cls(interactions, np.asarray(user_ids), np.asarray(item_ids))

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def has_timestamp(self) -> bool:
        return TIMESTAMP in self.interactions.columns

    def with_interactions(self, interactions: pd.DataFrame) -> InteractionData:
        """Same id space, different interaction subset (e.g. a train split)."""
        return InteractionData(interactions.reset_index(drop=True), self.user_ids, self.item_ids)

    def user_item_matrix(self) -> sp.csr_matrix:
        """Binary (n_users, n_items) matrix; repeated interactions count once."""
        rows = self.interactions[USER_IDX].to_numpy()
        cols = self.interactions[ITEM_IDX].to_numpy()
        matrix = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(self.n_users, self.n_items),
        )
        matrix.data[:] = 1.0
        return matrix

    def history_lengths(self) -> np.ndarray:
        """Number of interactions per user index."""
        return np.bincount(self.interactions[USER_IDX].to_numpy(), minlength=self.n_users)


@dataclass(frozen=True)
class EvaluationSplit:
    train: InteractionData
    eval_users: np.ndarray  # user indices with a held-out target
    targets: np.ndarray  # held-out item index per eval user


def leave_last_out(data: InteractionData, min_train_interactions: int) -> EvaluationSplit:
    """Hold out each user's last interaction (by timestamp, else row order).

    Users with fewer than `min_train_interactions + 1` interactions keep all their
    interactions in train and are not evaluated.
    """
    if min_train_interactions < 1:
        raise ValueError("min_train_interactions must be >= 1")
    frame = data.interactions.copy()
    frame[_ORDER] = np.arange(len(frame))
    sort_keys = [USER_IDX, TIMESTAMP, _ORDER] if data.has_timestamp else [USER_IDX, _ORDER]
    frame = frame.sort_values(sort_keys, kind="mergesort")

    by_user = frame.groupby(USER_IDX, sort=False)
    is_last = by_user.cumcount(ascending=False).to_numpy() == 0
    has_enough = by_user[ITEM_IDX].transform("size").to_numpy() > min_train_interactions
    is_test = is_last & has_enough

    test = frame[is_test]
    train = frame[~is_test].drop(columns=_ORDER)
    return EvaluationSplit(
        train=data.with_interactions(train),
        eval_users=test[USER_IDX].to_numpy(),
        targets=test[ITEM_IDX].to_numpy(),
    )


def load_interactions(dataset_config: dict[str, Any], rng: np.random.Generator) -> InteractionData:
    """Build `InteractionData` from a dataset config (`source: synthetic | csv`)."""
    source = dataset_config["source"]
    if source == "synthetic":
        frame = generate_synthetic_interactions(rng=rng, **dataset_config.get("params", {}))
        return InteractionData.from_frame(frame, timestamp_col=TIMESTAMP)
    if source == "csv":
        columns = {USER_ID: USER_ID, ITEM_ID: ITEM_ID, TIMESTAMP: None}
        columns.update(dataset_config.get("columns", {}))
        frame = pd.read_csv(dataset_config["path"])
        return InteractionData.from_frame(
            frame,
            user_col=columns[USER_ID],
            item_col=columns[ITEM_ID],
            timestamp_col=columns[TIMESTAMP],
        )
    raise ValueError(f"Unknown dataset source {source!r}; expected 'synthetic' or 'csv'")
