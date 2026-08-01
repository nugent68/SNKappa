"""Latent-variable (errors-in-variables) regression for the lensing amplitude.

Model, per supernova i in source-redshift bin b:

    xi_i  ~ N(0, tau_b^2)              latent catalog-visible convergence
    x_i | xi_i ~ N(xi_i, s_i^2)        observed prediction; s_i^2 = per-SN
                                       Monte-Carlo noise variance (photo-z,
                                       M*, SMHM, concentration, cluster
                                       mass/miscentering, zero point)
    y_i | xi_i ~ N(alpha + B xi_i, sigma_i^2),  B = -(5/ln10) A

Marginalizing xi analytically gives the heteroscedastic conditional

    y_i | x_i ~ N(alpha + B m_i,  B^2 v_i + sigma_i^2)
    m_i = w_i x_i,  v_i = w_i s_i^2,  w_i = tau_b^2 / (tau_b^2 + s_i^2)

i.e. each supernova is de-attenuated by its own reliability w_i -- the
"fully heteroscedastic latent-variable regression" of the manuscript.
tau_b^2 is set empirically per bin (empirical Bayes): the observed
variance of x in the bin minus the mean noise variance. Priors on the
latent field are Gaussian per bin; the skewness of the kappa field is
second order for the amplitude (it enters through shrinkage weights,
which depend on the prior only through tau) and is noted as a caveat.

The two-component version fits independent amplitudes for the
galaxy-halo and cluster tiers (their latent fields are treated as
independent, justified by the measured cross-correlation r = 0.015):

    y_i | x_g, x_c ~ N(alpha + B_g m_gi + B_c m_ci,
                       B_g^2 v_gi + B_c^2 v_ci + sigma_i^2)

alpha is profiled analytically (it is a weighted mean); A posteriors are
evaluated on a grid with a flat prior; quoted errors are the posterior
std with an outer bootstrap over supernovae (which re-estimates tau_b)
added in quadrature.
"""

from __future__ import annotations

import numpy as np

SLOPE_TH = -5.0 / np.log(10.0)
TAU2_FLOOR = 1e-10
MIN_BIN = 20   # bins with fewer SNe fall back to the randoms-based tau


def bin_tau2(x, s2, zbin, rand_based_tau2=None):
    """Empirical-Bayes tau_b^2 per z bin: Var_b(x) - <s^2>_b, floored.

    rand_based_tau2: optional dict {zbin: tau2} fallback for thin bins
    (e.g. from the random-sightline scatter minus the mean noise).
    Returns tau2 aligned with x (per-SN).
    """
    x = np.asarray(x, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    zbin = np.asarray(zbin)
    tau2 = np.empty_like(x)
    for zb in np.unique(zbin):
        m = zbin == zb
        est = x[m].var() - s2[m].mean()
        if m.sum() < MIN_BIN and rand_based_tau2 is not None:
            est = rand_based_tau2.get(zb, est)
        tau2[m] = max(est, TAU2_FLOOR)
    return tau2


def _profile_alpha_loglike(mu_pred, var_tot, y):
    """log L profiled over the intercept alpha (analytic weighted mean)."""
    w = 1.0 / var_tot
    alpha = np.sum(w * (y - mu_pred)) / np.sum(w)
    r = y - mu_pred - alpha
    return -0.5 * np.sum(r * r * w + np.log(var_tot)), alpha


def loglike_amplitude(A, x, s2, tau2, y, sig2_y, s2_berk=0.0):
    """Profiled log-likelihood of the single amplitude A.

    s2 is the CLASSICAL noise variance (prediction built from noisy
    measurements: M* measurement error, richness-mass error) -- it drives
    the shrinkage. s2_berk is the BERKSON variance (uncertainty the
    prediction already marginalizes: photo-z p(z), SMHM and concentration
    intrinsic scatter, miscentering) -- it adds B^2 s2_berk to the
    residual and does NOT attenuate; treating it as classical over-shrinks
    the noisiest sight lines and inflates A.
    """
    B = SLOPE_TH * A
    w = tau2 / (tau2 + s2)
    m = w * x
    v = w * s2
    return _profile_alpha_loglike(
        B * m, B * B * (v + s2_berk) + sig2_y, y)[0]


def loglike_two(Ag, Ac, xg, s2g, tau2g, xc, s2c, tau2c, y, sig2_y,
                s2g_berk=0.0, s2c_berk=0.0):
    """Profiled log-likelihood of (A_gal, A_cl); classical/Berkson split
    per component as in loglike_amplitude."""
    Bg = SLOPE_TH * Ag
    Bc = SLOPE_TH * Ac
    wg = tau2g / (tau2g + s2g)
    wc = tau2c / (tau2c + s2c)
    mu = Bg * wg * xg + Bc * wc * xc
    var = (Bg * Bg * (wg * s2g + s2g_berk)
           + Bc * Bc * (wc * s2c + s2c_berk) + sig2_y)
    return _profile_alpha_loglike(mu, var, y)[0]


def _posterior_moments(grid, logl):
    logl = logl - logl.max()
    p = np.exp(logl)
    p /= np.trapezoid(p, grid)
    mean = np.trapezoid(grid * p, grid)
    var = np.trapezoid((grid - mean) ** 2 * p, grid)
    return float(mean), float(np.sqrt(var))


def fit_amplitude(x, s2, zbin, y, sig2_y, rng=None, n_boot=200,
                  grid=None, rand_based_tau2=None, s2_berk=None):
    """Posterior mean/err of A (flat prior), with bootstrap tau_b noise.

    s2 = classical noise variance (drives shrinkage); s2_berk = Berkson
    variance (adds to the residual only). Returns dict with A, A_err
    (posterior std and bootstrap added in quadrature), and the mean
    reliability <w>.
    """
    x = np.asarray(x, float); s2 = np.asarray(s2, float)
    y = np.asarray(y, float); sig2_y = np.asarray(sig2_y, float)
    zbin = np.asarray(zbin)
    sb = 0.0 if s2_berk is None else np.asarray(s2_berk, float)
    grid = np.linspace(-1.0, 3.0, 401) if grid is None else grid

    tau2 = bin_tau2(x, s2, zbin, rand_based_tau2)
    logl = np.array([loglike_amplitude(a, x, s2, tau2, y, sig2_y, sb)
                     for a in grid])
    A, sA = _posterior_moments(grid, logl)

    boot_sd = 0.0
    if rng is not None and n_boot:
        n = x.size
        boots = np.empty(n_boot)
        sb_arr = np.broadcast_to(sb, x.shape)
        for k in range(n_boot):
            i = rng.integers(0, n, n)
            t2 = bin_tau2(x[i], s2[i], zbin[i], rand_based_tau2)
            ll = np.array([loglike_amplitude(a, x[i], s2[i], t2, y[i],
                                             sig2_y[i], sb_arr[i])
                           for a in grid])
            boots[k] = _posterior_moments(grid, ll)[0]
        boot_sd = float(boots.std())

    return {"A": A, "A_err_post": sA, "A_err_boot": boot_sd,
            "A_err": float(np.hypot(sA, boot_sd)),
            "mean_reliability": float(np.mean(tau2 / (tau2 + s2)))}


def fit_two_component(xg, s2g, xc, s2c, zbin, y, sig2_y, rng=None,
                      n_boot=100, grid=None, s2g_berk=None, s2c_berk=None):
    """Joint posterior for (A_gal, A_cl) on a grid; flat priors.

    Classical/Berkson split per component as in fit_amplitude. Returns
    dict with marginal means/errors and the posterior correlation.
    """
    xg = np.asarray(xg, float); s2g = np.asarray(s2g, float)
    xc = np.asarray(xc, float); s2c = np.asarray(s2c, float)
    y = np.asarray(y, float); sig2_y = np.asarray(sig2_y, float)
    zbin = np.asarray(zbin)
    sbg = 0.0 if s2g_berk is None else np.asarray(s2g_berk, float)
    sbc = 0.0 if s2c_berk is None else np.asarray(s2c_berk, float)
    grid = np.linspace(-1.0, 3.0, 161) if grid is None else grid

    t2g = bin_tau2(xg, s2g, zbin)
    t2c = bin_tau2(xc, s2c, zbin)
    ll = np.array([[loglike_two(ag, ac, xg, s2g, t2g, xc, s2c, t2c,
                                y, sig2_y, sbg, sbc)
                    for ac in grid] for ag in grid])
    ll -= ll.max()
    p = np.exp(ll)
    p /= p.sum()
    pg = p.sum(axis=1); pc = p.sum(axis=0)
    Ag = float(np.sum(grid * pg)); Ac = float(np.sum(grid * pc))
    sg = float(np.sqrt(np.sum((grid - Ag) ** 2 * pg)))
    sc = float(np.sqrt(np.sum((grid - Ac) ** 2 * pc)))
    cov = float(np.sum(p * np.outer(grid - Ag, grid - Ac)))

    boot_g = boot_c = 0.0
    if rng is not None and n_boot:
        n = xg.size
        bg = np.empty(n_boot); bc = np.empty(n_boot)
        sbg_a = np.broadcast_to(sbg, xg.shape)
        sbc_a = np.broadcast_to(sbc, xc.shape)
        coarse = grid[::4] if grid.size > 80 else grid
        for k in range(n_boot):
            i = rng.integers(0, n, n)
            tg = bin_tau2(xg[i], s2g[i], zbin[i])
            tc = bin_tau2(xc[i], s2c[i], zbin[i])
            llb = np.array([[loglike_two(a1, a2, xg[i], s2g[i], tg,
                                         xc[i], s2c[i], tc, y[i],
                                         sig2_y[i], sbg_a[i], sbc_a[i])
                             for a2 in coarse] for a1 in coarse])
            llb -= llb.max()
            pb = np.exp(llb); pb /= pb.sum()
            bg[k] = np.sum(coarse * pb.sum(axis=1))
            bc[k] = np.sum(coarse * pb.sum(axis=0))
        boot_g = float(bg.std()); boot_c = float(bc.std())

    return {"A_gal": Ag, "A_gal_err": float(np.hypot(sg, boot_g)),
            "A_cl": Ac, "A_cl_err": float(np.hypot(sc, boot_c)),
            "corr": cov / (sg * sc) if sg * sc > 0 else 0.0}
