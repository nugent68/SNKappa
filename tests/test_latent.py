"""Latent-variable regression: the EIV fit must recover the true
amplitude where a naive weighted fit is attenuated."""

import numpy as np

from snkappa.fitting import bootstrap_slope
from snkappa.latent import SLOPE_TH, fit_amplitude, fit_two_component


def _make_data(rng, n=3000, a_true=1.0):
    zbin = rng.integers(0, 10, n)
    tau = 0.004 + 0.0008 * zbin            # signal grows with z
    xi = rng.standard_normal(n) * tau
    s = rng.uniform(0.002, 0.008, n)       # heteroscedastic pred noise
    x = xi + rng.standard_normal(n) * s
    sig_y = rng.uniform(0.10, 0.25, n)
    y = SLOPE_TH * a_true * xi + rng.standard_normal(n) * sig_y + 0.02
    return x, s ** 2, zbin, y, sig_y ** 2, xi


def test_latent_recovers_where_naive_attenuates():
    rng = np.random.default_rng(11)
    x, s2, zbin, y, sig2y, xi = _make_data(rng, a_true=1.0)

    # naive weighted slope is attenuated by lambda = Var(xi)/Var(x) ~ 0.5
    b, _ = bootstrap_slope(x, y, np.sqrt(sig2y), rng, n_boot=200)
    lam = xi.var() / x.var()
    assert b / SLOPE_TH < 0.75             # visibly attenuated
    assert abs(b / SLOPE_TH - lam) < 0.2   # ... by roughly lambda

    fit = fit_amplitude(x, s2, zbin, y, sig2y, rng=rng, n_boot=50)
    assert abs(fit["A"] - 1.0) < 3 * fit["A_err"]
    assert fit["A_err"] < 0.45
    assert 0.3 < fit["mean_reliability"] < 0.8


def test_berkson_noise_must_not_shrink():
    """When the prediction MARGINALIZES an uncertainty (Berkson error:
    truth = x + e with e independent of x), the naive slope is unbiased
    and the latent fit with the correct classical/Berkson split must not
    de-attenuate; misclassifying Berkson variance as classical inflates
    A. This is the failure mode the split exists to prevent."""
    rng = np.random.default_rng(5)
    n = 4000
    zbin = rng.integers(0, 10, n)
    x = rng.standard_normal(n) * 0.006          # prediction (no classical err)
    e_b = rng.standard_normal(n) * 0.004        # marginalized scatter
    kappa_true = x + e_b
    sig_y = rng.uniform(0.10, 0.2, n)
    y = SLOPE_TH * 1.0 * kappa_true + rng.standard_normal(n) * sig_y

    s2_tiny = np.full(n, 1e-10)
    s2_b = np.full(n, 0.004 ** 2)

    ok = fit_amplitude(x, s2_tiny, zbin, y, sig_y ** 2, s2_berk=s2_b)
    assert abs(ok["A"] - 1.0) < 3 * ok["A_err_post"]

    wrong = fit_amplitude(x, s2_b, zbin, y, sig_y ** 2)  # misclassified
    assert wrong["A"] > ok["A"] + 0.2           # visibly inflated


def test_latent_two_component_recovery():
    rng = np.random.default_rng(23)
    n = 3000
    zbin = rng.integers(0, 10, n)
    tau_g = 0.004 + 0.0005 * zbin
    tau_c = 0.006 + 0.0008 * zbin
    xg_t = rng.standard_normal(n) * tau_g
    xc_t = rng.standard_normal(n) * tau_c
    sg = rng.uniform(0.002, 0.005, n)
    sc = rng.uniform(0.003, 0.009, n)      # cluster tier noisier
    xg = xg_t + rng.standard_normal(n) * sg
    xc = xc_t + rng.standard_normal(n) * sc
    sig_y = rng.uniform(0.10, 0.25, n)
    y = SLOPE_TH * (0.9 * xg_t + 1.2 * xc_t) \
        + rng.standard_normal(n) * sig_y
    fit = fit_two_component(xg, sg ** 2, xc, sc ** 2, zbin, y, sig_y ** 2,
                            rng=rng, n_boot=30)
    assert abs(fit["A_gal"] - 0.9) < 3 * fit["A_gal_err"]
    assert abs(fit["A_cl"] - 1.2) < 3 * fit["A_cl_err"]
    assert abs(fit["corr"]) < 0.5          # independent latents stay so
