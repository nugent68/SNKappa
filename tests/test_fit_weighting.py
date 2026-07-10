"""Regression test for the np.polyfit weight convention (TODO 0.1).

np.polyfit minimizes sum (w_i r_i)^2, so inverse-variance weighting needs
w = 1/sigma. The old code passed w = 1/sigma^2 (effective sigma^-4 weights).
The weighted slope from snkappa.fitting must match the analytic weighted
least-squares solution with weights 1/sigma^2.
"""

import numpy as np

from snkappa.fitting import weighted_slope, bootstrap_slope


def analytic_wls(x, y, sigma):
    """Closed-form WLS slope/intercept with weights 1/sigma^2."""
    w = 1.0 / sigma**2
    sw = w.sum()
    xb = (w * x).sum() / sw
    yb = (w * y).sum() / sw
    b = (w * (x - xb) * (y - yb)).sum() / (w * (x - xb) ** 2).sum()
    a = yb - b * xb
    return b, a


def make_data(rng, n=800):
    """Heteroscedastic synthetic data: sigma spans a factor ~8."""
    x = rng.uniform(-0.02, 0.08, n)
    sigma = rng.uniform(0.05, 0.4, n)
    y = -2.17 * x + 0.01 + sigma * rng.standard_normal(n)
    return x, y, sigma


def test_weighted_slope_matches_analytic_wls():
    rng = np.random.default_rng(7)
    x, y, sigma = make_data(rng)
    b, a = weighted_slope(x, y, sigma)
    b_ref, a_ref = analytic_wls(x, y, sigma)
    assert abs(b - b_ref) < 1e-8 * max(1.0, abs(b_ref))
    assert abs(a - a_ref) < 1e-10


def test_wrong_convention_differs():
    """Guard: with heteroscedastic noise, the old 1/sigma^2 polyfit weights
    give a measurably different slope, so this test would have caught it."""
    rng = np.random.default_rng(11)
    x, y, sigma = make_data(rng)
    b_ref, _ = analytic_wls(x, y, sigma)
    b_wrong = np.polyfit(x, y, 1, w=1.0 / sigma**2)[0]
    assert abs(b_wrong - b_ref) > 1e-3


def test_bootstrap_slope_consistent():
    rng = np.random.default_rng(13)
    x, y, sigma = make_data(rng, n=1500)
    b, e = bootstrap_slope(x, y, sigma, rng, n_boot=400)
    b_ref, _ = analytic_wls(x, y, sigma)
    assert abs(b - b_ref) < 1e-8 * max(1.0, abs(b_ref))
    # true slope -2.17 should be recovered within ~4 bootstrap sigma
    assert abs(b - (-2.17)) < 4 * e
