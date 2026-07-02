"""Monte Carlo propagation to P(kappa_ext).

Jointly resamples, per draw and per galaxy in the SN aperture:
  - photo-z from its split-normal p(z) (draws at z >= z_src contribute zero),
  - M*/L scatter (mstar_scatter_dex),
  - SMHM scatter (smhm_scatter_dex),
  - concentration scatter (c_scatter_dex).
The final P(kappa_ext) subtracts the random-sightline mean (mass-sheet zero
point) and adds the empirical random-LOS scatter in quadrature, treated as the
systematic floor for LOS structure the catalog cannot see.
"""

from __future__ import annotations

import numpy as np


def mc_kappa_raw(cfg, engine, rng, exclude_mask):
    """Draws of kappa_raw for the SN sightline; returns array [n_mc]."""
    hm = engine.hm
    n_mc = cfg.montecarlo.n_mc
    r_ap = cfg.los.aperture_radius_arcmin * 60.0
    r_in = cfg.deflector.r_exclude_arcsec

    from .kappa import angular_sep_arcsec
    theta_all = angular_sep_arcsec(
        cfg.source.ra_src, cfg.source.dec_src,
        engine.df["ra"].to_numpy(), engine.df["dec"].to_numpy())
    sel = (theta_all > r_in) & (theta_all < r_ap) & ~exclude_mask

    # ---- galaxies and their per-draw redshifts -----------------------------
    spec_in = sel[engine.spec_idx]
    phot_in = sel[engine.phot_idx]
    gi_spec = engine.spec_idx[spec_in]
    gi_phot = engine.phot_idx[phot_in]
    n_s, n_p = gi_spec.size, gi_phot.size

    from . import photoz
    z_phot_draws = photoz.sample(
        rng, engine.phot_mu[phot_in], engine.phot_slo[phot_in],
        engine.phot_shi[phot_in], n_mc)                     # [n_mc, n_p]

    # spec-z fixed across draws
    z_spec = engine.df["z_spec"].to_numpy()[gi_spec]
    z = np.concatenate([np.broadcast_to(z_spec, (n_mc, n_s)),
                        z_phot_draws], axis=1)              # [n_mc, n_gal]
    theta = np.concatenate([theta_all[gi_spec], theta_all[gi_phot]])
    n_gal = n_s + n_p

    mags = {b: np.concatenate([
        engine.df[f"mag_{b}"].to_numpy()[gi_spec],
        engine.df[f"mag_{b}"].to_numpy()[gi_phot]])
        for b in "griz" if f"mag_{b}" in engine.df}

    # ---- scatter draws ------------------------------------------------------
    hcfg = cfg.halo_model
    d_ml = rng.normal(0.0, hcfg.mstar_scatter_dex, (n_mc, n_gal))
    d_smhm = rng.normal(0.0, hcfg.smhm_scatter_dex, (n_mc, n_gal))
    d_c = rng.normal(0.0, hcfg.c_scatter_dex, (n_mc, n_gal))

    foreground = z < cfg.source.z_src
    ibin = hm.zbin_index(z)

    kappa_draws = np.zeros(n_mc)
    flat_bin = ibin.ravel()
    flat_fg = foreground.ravel()
    flat_ml = d_ml.ravel()
    flat_smhm = d_smhm.ravel()
    flat_c = d_c.ravel()
    flat_theta = np.broadcast_to(theta, (n_mc, n_gal)).ravel()
    draw_of = np.broadcast_to(np.arange(n_mc)[:, None], (n_mc, n_gal)).ravel()
    flat_mags = {b: np.broadcast_to(v, (n_mc, n_gal)).ravel()
                 for b, v in mags.items()}

    # group all (draw, galaxy) pairs by z bin; vectorize within each bin
    for b in np.unique(flat_bin[flat_fg]):
        m = flat_fg & (flat_bin == b)
        zb = hm.zbins[b]
        logms = engine.stellar.logmstar(
            {k: v[m] for k, v in flat_mags.items()},
            np.full(m.sum(), zb)) + flat_ml[m]
        rhos, rs, tau = hm.halo_params(
            logms, np.full(m.sum(), b), dlogm=flat_smhm[m], dlogc=flat_c[m])
        amp = rhos * rs / engine.sigcr[b]
        x = flat_theta[m] / (rs / hm.da[b] * 206264.806)
        k = amp * hm.sigma_dimless(x, tau)
        np.add.at(kappa_draws, draw_of[m], k)

    return kappa_draws


def build_pkappa(cfg, kappa_raw_draws, randoms, rng):
    """Final P(kappa_ext) samples: zero-point subtraction + empirical LOS
    scatter convolution."""
    mu_rand = randoms["kappa_mean"]
    sig_rand = randoms["kappa_std"]
    n = kappa_raw_draws.size
    return (kappa_raw_draws - mu_rand
            + rng.normal(0.0, sig_rand, n))


def percentiles(samples, levels=(2.5, 16, 50, 84, 97.5)):
    return {f"p{lev:g}": float(np.percentile(samples, lev)) for lev in levels}
