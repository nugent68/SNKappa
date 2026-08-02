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


def test_two_component_recovers_distinct_slopes():
    """Synthetic heteroscedastic data with different slopes on two
    correlated regressors must be recovered by the WLS fit."""
    import numpy as np
    from snkappa.fitting import two_component_slopes

    rng = np.random.default_rng(7)
    n = 4000
    x1 = rng.normal(0, 0.01, n)
    x2 = 0.3 * x1 + rng.normal(0, 0.004, n)   # correlated components
    sigma = rng.uniform(0.08, 0.3, n)
    y = -2.0 * x1 - 4.0 * x2 + 0.01 + rng.standard_normal(n) * sigma
    (b1, e1), (b2, e2) = two_component_slopes(x1, x2, y, sigma, rng)
    assert abs(b1 - (-2.0)) < 3 * e1
    assert abs(b2 - (-4.0)) < 3 * e2
    assert e1 < 1.0 and e2 < 2.5


def test_gls_slope_correlated_noise():
    """GLS must recover the slope AND the correct uncertainty under
    block-correlated noise, where the diagonal fit's error is wrong;
    with a diagonal matrix it must match the weighted fit exactly."""
    import numpy as np
    from snkappa.fitting import bootstrap_slope, gls_slope

    rng = np.random.default_rng(9)
    n, n_real = 400, 300
    x = rng.normal(0, 0.01, n)
    block = rng.integers(0, 20, n)          # 20 correlated survey blocks
    sig = rng.uniform(0.1, 0.2, n)
    C = np.diag(sig**2)
    for bl in range(20):
        m = np.flatnonzero(block == bl)
        for i in m:
            for j in m:
                if i != j:
                    C[i, j] = 0.5 * sig[i] * sig[j]
    L = np.linalg.cholesky(C)
    b_true = -2.0
    ests, errs = [], []
    for _ in range(n_real):
        y = b_true * x + L @ rng.standard_normal(n)
        b, e = gls_slope(x, y, C)
        ests.append(b); errs.append(e)
    ests = np.array(ests)
    # unbiased and correctly calibrated error
    assert abs(ests.mean() - b_true) < 3 * ests.std() / np.sqrt(n_real)
    assert abs(np.mean(errs) / ests.std() - 1.0) < 0.15

    # diagonal C reproduces the weighted (1/sigma) polyfit slope
    y = b_true * x + rng.standard_normal(n) * sig
    b_g, _ = gls_slope(x, y, np.diag(sig**2))
    b_w = np.polyfit(x, y, 1, w=1.0 / sig)[0]
    assert abs(b_g - b_w) < 1e-10
