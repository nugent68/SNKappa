"""Stellar masses from LS photometry: color-based M*/L (Taylor et al. 2011).

Default estimator (swappable via the StellarMassEstimator interface):

    log10(M*/Msun) = 1.15 + 0.70 (g-i)_rest - 0.4 M_i   [Taylor+2011, eq. 8]

Approximations (deliberate, documented; their scatter is folded into the Monte
Carlo via halo_model.mstar_scatter_dex):
- K-correction: flat-Fnu SED, K = -2.5 log10(1+z) applied to the luminosity
  band; observed color is used as a proxy for rest-frame color.
- LS north (BASS/MzLS) has no i-band: we use (g-z) with the color coefficient
  rescaled by the mean (g-i)/(g-z) ratio of galaxy SEDs (~0.78), and the
  observed z-band as the luminosity band.

A full per-galaxy SED fit (e.g. CIGALE/Prospector) can replace this by
implementing StellarMassEstimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

LOGM_MIN, LOGM_MAX = 7.0, 12.2  # clip to avoid absurd halos from bad photometry


class StellarMassEstimator(ABC):
    @abstractmethod
    def logmstar(self, mags: dict, z: np.ndarray) -> np.ndarray:
        """log10 stellar mass [Msun]; mags = dict of dereddened AB mags."""


class Taylor2011(StellarMassEstimator):
    """Color-based M*/L; (g-i) in the DECam south, (g-z) in the north."""

    def __init__(self, cosmo):
        self.cosmo = cosmo

    def logmstar(self, mags, z):
        z = np.asarray(z, dtype=float)
        g = np.asarray(mags["g"], dtype=float)
        i = np.asarray(mags.get("i", np.full_like(g, np.nan)), dtype=float)
        zb = np.asarray(mags["z"], dtype=float)

        dm = self.cosmo.distmod(np.clip(z, 1e-3, None)).value
        kcorr = -2.5 * np.log10(1.0 + z)  # flat-Fnu approximation

        # clip colors to (a slightly padded version of) the Taylor+2011
        # calibration range: extreme colors are photometric junk / heavy dust
        # and would otherwise blow up the M*/L exponentially
        col_gi = np.clip(g - i, 0.0, 2.2)
        col_gz = np.clip(g - zb, 0.0, 3.0)

        # south: Taylor+2011 as published
        logm_south = 1.15 + 0.70 * col_gi - 0.4 * (i - dm - kcorr)
        # north: g-z color, coefficient rescaled; z-band luminosity
        logm_north = 1.15 + 0.55 * col_gz - 0.4 * (zb - dm - kcorr)

        logm = np.where(np.isfinite(logm_south), logm_south, logm_north)
        bad = ~np.isfinite(logm)
        # last resort: pure luminosity scaling with a solar-ish M*/L
        logm = np.where(bad, 1.15 + 0.7 - 0.4 * (zb - dm - kcorr), logm)
        return np.clip(logm, LOGM_MIN, LOGM_MAX)


def make_estimator(name: str, cosmo) -> StellarMassEstimator:
    if name == "taylor2011":
        return Taylor2011(cosmo)
    raise ValueError(f"Unknown mstar_method: {name}")
