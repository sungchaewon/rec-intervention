"""Item category side information (e.g. movie genres, product categories)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

ITEM_ID = "item_id"
CATEGORY = "category"


@dataclass(frozen=True)
class ItemCategories:
    """`distribution[i]` is p(category | item i): an item with g categories puts 1/g
    on each (Steck 2018). Items without categories have an all-zero row.
    Rows follow the item index of the `InteractionData` it was built for."""

    distribution: sp.csr_matrix  # (n_items, n_categories)
    names: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, item_ids: np.ndarray) -> ItemCategories:
        """`frame` is long-format (item_id, category); unknown items are ignored."""
        item_index = pd.Index(item_ids).get_indexer(frame[ITEM_ID])
        known = item_index >= 0
        codes, names = pd.factorize(frame.loc[known, CATEGORY], sort=True)
        membership = sp.csr_matrix(
            (np.ones(len(codes)), (item_index[known], codes)), shape=(len(item_ids), len(names))
        )
        membership.data[:] = 1.0  # duplicate (item, category) rows count once
        counts = np.asarray(membership.sum(axis=1)).ravel()
        inverse = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
        return cls(sp.diags(inverse) @ membership, np.asarray(names))

    @property
    def n_categories(self) -> int:
        return len(self.names)
