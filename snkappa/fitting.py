"""Weighted regression helpers shared by the batch scripts and the paper.

np.polyfit's weight convention: w multiplies the UNSQUARED residual, i.e. it
minimizes sum_i (w_i (y_i - p(x_i)))^2. Inverse-variance weighting therefore
requires w = 1/sigma (NOT 1/sigma^2, which silently applies sigma^-4 weights
and over-weights the lowest-error points).
"""

from __future__ import annotations

import numpy as np


def weighted_slope(x, y, sigma):
    """Inverse-variance weighted linear fit; returns (slope, intercept)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    b, a = np.polyfit(x, y, 1, w=1.0 / sigma)
    return float(b), float(a)


def bootstrap_slope(x, y, sigma, rng, n_boot=2000):
    """(slope, bootstrap error) for the inverse-variance weighted fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = 1.0 / np.asarray(sigma, dtype=float)
    b = float(np.polyfit(x, y, 1, w=w)[0])
    boot = np.empty(n_boot)
    n = x.size
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        boot[k] = np.polyfit(x[i], y[i], 1, w=w[i])[0]
    return b, float(boot.std())
