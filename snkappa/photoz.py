"""Photometric-redshift PDFs reconstructed from served quantiles.

The Data Lab photo-z tables (ls_dr10.photo_z / ls_dr9.photo_z, Zhou et al. 2021
random-forest photo-z) serve point estimates and 68/95% quantiles but NOT the
full p(z). We model p(z) as a two-piece (split) Gaussian pinned to
(z_phot_median, z_phot_l68, z_phot_u68), truncated at z > 0. This is recorded
as a provenance deviation.
"""

from __future__ import annotations

import numpy as np

_MIN_SIGMA = 0.01  # floor on either side width, in redshift units


def split_normal_params(z_med, z_l68, z_u68, z_std=None):
    """Per-galaxy (mu, sigma_lo, sigma_hi) arrays with sane fallbacks."""
    z_med = np.asarray(z_med, dtype=float)
    sig_lo = z_med - np.asarray(z_l68, dtype=float)
    sig_hi = np.asarray(z_u68, dtype=float) - z_med
    if z_std is not None:
        z_std = np.asarray(z_std, dtype=float)
        bad_lo = ~np.isfinite(sig_lo) | (sig_lo <= 0)
        bad_hi = ~np.isfinite(sig_hi) | (sig_hi <= 0)
        sig_lo = np.where(bad_lo, z_std, sig_lo)
        sig_hi = np.where(bad_hi, z_std, sig_hi)
    sig_lo = np.clip(np.nan_to_num(sig_lo, nan=_MIN_SIGMA), _MIN_SIGMA, None)
    sig_hi = np.clip(np.nan_to_num(sig_hi, nan=_MIN_SIGMA), _MIN_SIGMA, None)
    return z_med, sig_lo, sig_hi


def grid_pdf(zgrid, mu, sig_lo, sig_hi):
    """p(z) evaluated on zgrid for each galaxy; shape [n_gal, n_z].

    Model: two half-Gaussians joined at mu, each carrying probability mass 0.5,
    so that mu IS the median and (mu - sig_lo, mu + sig_hi) are exactly the
    16th/84th percentiles -- i.e. the served photo-z quantiles are pinned
    exactly. (The density has a small step at mu when sig_lo != sig_hi; the
    quantile pinning is what matters for lensing weights.)

    Normalized so that the integral over the FULL z>0 axis is 1 -- weights on a
    truncated grid (z < z_src) then correctly down-weight galaxies whose PDF
    leaks past the source redshift.
    """
    zgrid = np.asarray(zgrid, dtype=float)
    mu = np.atleast_1d(mu)[:, None]
    slo = np.atleast_1d(sig_lo)[:, None]
    shi = np.atleast_1d(sig_hi)[:, None]
    z = zgrid[None, :]
    sig = np.where(z < mu, slo, shi)
    # each side: half-Gaussian with mass 0.5 -> amplitude 1/(sqrt(2pi) sigma)
    pdf = np.exp(-0.5 * ((z - mu) / sig) ** 2) / (np.sqrt(2.0 * np.pi) * sig)
    from scipy.special import erf
    # mass at z<0 (low side only): Phi(-mu/slo), Phi(t) = 0.5(1+erf(t/sqrt2))
    p_neg = 0.5 * (1.0 + erf(-mu / (np.sqrt(2.0) * slo)))
    pdf = pdf / np.clip(1.0 - p_neg, 1e-6, None)
    pdf[:, zgrid <= 0] = 0.0
    return pdf


def grid_weights(zgrid, mu, sig_lo, sig_hi):
    """Integration weights w[g, i] ~ p_g(z_i) * dz_i on the (truncated) grid."""
    zgrid = np.asarray(zgrid, dtype=float)
    dz = np.gradient(zgrid)
    return grid_pdf(zgrid, mu, sig_lo, sig_hi) * dz[None, :]


def sample(rng, mu, sig_lo, sig_hi, size):
    """Draw z samples from each galaxy's two-half-Gaussian p(z);
    shape [size, n_gal]. Each side carries mass 0.5 (median = mu).

    Draws with z<=0 are resampled once, then clipped to a small positive value
    (negligible probability mass by construction).
    """
    mu = np.atleast_1d(mu)
    slo = np.atleast_1d(sig_lo)
    shi = np.atleast_1d(sig_hi)
    n = mu.size
    for _ in range(2):
        pick_lo = rng.random((size, n)) < 0.5
        mag = np.abs(rng.standard_normal((size, n)))
        z = np.where(pick_lo, mu[None, :] - mag * slo[None, :],
                     mu[None, :] + mag * shi[None, :])
        if (z > 0).all():
            break
    return np.clip(z, 1e-3, None)
