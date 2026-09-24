from __future__ import annotations

import numpy as np

# Resamples drawn per vectorized batch; bounds memory at batch * n_samples ints.
_RESAMPLE_BATCH_SIZE = 256


def bootstrap_mean_ci(
    values: np.ndarray,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")

    means = np.empty(n_resamples)
    for start in range(0, n_resamples, _RESAMPLE_BATCH_SIZE):
        stop = min(start + _RESAMPLE_BATCH_SIZE, n_resamples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)

    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return float(low), float(high)
