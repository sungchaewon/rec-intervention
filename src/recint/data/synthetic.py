"""Synthetic implicit-feedback data for smoke tests and pipeline development.

Each item belongs to a latent cluster and has a Zipf-like popularity. Each user
prefers one cluster and draws a variable-length history (log-normal length, so
history sparsity varies across users) without replacement, mixing in-cluster
and popularity-driven choices. Not intended to model any real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_interactions(
    n_users: int,
    n_items: int,
    n_clusters: int,
    min_history: int,
    max_history: int,
    history_log_mean: float,
    history_log_std: float,
    popularity_exponent: float,
    in_cluster_prob: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if not 1 <= min_history <= max_history < n_items:
        raise ValueError("Require 1 <= min_history <= max_history < n_items")
    if not 0.0 < in_cluster_prob < 1.0:
        raise ValueError("in_cluster_prob must be in (0, 1)")
    if not 2 <= n_clusters <= n_items:
        raise ValueError("Require 2 <= n_clusters <= n_items")

    # Every cluster gets at least one item, so in/out-of-cluster masses are nonzero.
    item_cluster = rng.permutation(np.arange(n_items) % n_clusters)
    popularity_rank = rng.permutation(n_items)
    popularity = 1.0 / (popularity_rank + 1.0) ** popularity_exponent

    user_cluster = rng.integers(0, n_clusters, size=n_users)
    lengths = np.clip(
        np.round(rng.lognormal(history_log_mean, history_log_std, size=n_users)),
        min_history,
        max_history,
    ).astype(int)

    user_ids, item_ids, timestamps = [], [], []
    for user, (cluster, length) in enumerate(zip(user_cluster, lengths)):
        in_cluster = item_cluster == cluster
        weights = np.where(
            in_cluster,
            in_cluster_prob * popularity / popularity[in_cluster].sum(),
            (1.0 - in_cluster_prob) * popularity / popularity[~in_cluster].sum(),
        )
        items = rng.choice(n_items, size=length, replace=False, p=weights / weights.sum())
        user_ids.append(np.full(length, user))
        item_ids.append(items)
        timestamps.append(np.arange(length))

    return pd.DataFrame(
        {
            "user_id": np.concatenate(user_ids),
            "item_id": np.concatenate(item_ids),
            "timestamp": np.concatenate(timestamps),
        }
    )
