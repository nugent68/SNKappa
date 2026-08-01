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


def two_component_slopes(x1, x2, y, sigma, rng, n_boot=2000):
    """Weighted fit y = b1*x1 + b2*x2 + c (rows scaled by 1/sigma, i.e.
    inverse-variance WLS); returns ((b1, err1), (b2, err2)) with bootstrap
    errors. Used to fit separate lensing amplitudes for the galaxy-halo
    and cluster tiers of the convergence prediction."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y = np.asarray(y, dtype=float)
    w = 1.0 / np.asarray(sigma, dtype=float)
    X = np.column_stack([x1, x2, np.ones_like(x1)])

    def solve(Xs, ys, ws):
        beta, *_ = np.linalg.lstsq(Xs * ws[:, None], ys * ws, rcond=None)
        return beta

    b = solve(X, y, w)
    n = y.size
    boot = np.empty((n_boot, 2))
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        boot[k] = solve(X[i], y[i], w[i])[:2]
    return ((float(b[0]), float(boot[:, 0].std())),
            (float(b[1]), float(boot[:, 1].std())))


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
