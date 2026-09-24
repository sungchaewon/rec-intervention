"""Discretize continuous user-state values into named groups."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

GROUPING_METHODS = ("quantile", "fixed")


def assign_state_groups(
    values: np.ndarray,
    method: str,
    cut_points: Sequence[float],
    labels: Sequence[str],
) -> np.ndarray:
    """Assign each value to a group label.

    `cut_points` are quantile levels in (0, 1) for `method="quantile"` (computed
    over `values`) or raw state values for `method="fixed"`. Group i contains
    values in (edge[i-1], edge[i]], i.e. a value equal to an edge goes to the lower
    group. With heavy ties, quantile edges may coincide and leave groups empty.
    """
    if len(labels) != len(cut_points) + 1:
        raise ValueError(f"Need len(labels) == len(cut_points) + 1, got {len(labels)} labels")
    if list(cut_points) != sorted(cut_points):
        raise ValueError("cut_points must be sorted ascending")
    if method == "quantile":
        if not all(0.0 < q < 1.0 for q in cut_points):
            raise ValueError("Quantile cut_points must be in (0, 1)")
        edges = np.quantile(values, cut_points)
    elif method == "fixed":
        edges = np.asarray(cut_points, dtype=np.float64)
    else:
        raise ValueError(f"Unknown grouping method {method!r}; expected one of {GROUPING_METHODS}")
    group_index = np.searchsorted(edges, values, side="left")
    return np.asarray(labels, dtype=object)[group_index]
