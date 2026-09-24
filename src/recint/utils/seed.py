from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed global RNGs (python, numpy, torch if installed).

    Library code should prefer explicit `np.random.Generator`s from `make_rngs`;
    this exists for third-party code that relies on global state.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def make_rngs(seed: int, names: list[str]) -> dict[str, np.random.Generator]:
    """Independent, reproducible generators per pipeline stage.

    Adding a new stage at the end of `names` does not change existing streams.
    """
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children)}
