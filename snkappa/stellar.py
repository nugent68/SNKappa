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


class W1Taylor(StellarMassEstimator):
    """WISE W1 luminosity-based stellar mass, optical-color fallback.

    Rest-frame 3.4 um light is a near-SED-independent stellar-mass tracer:
        M*/L_W1 ~ 0.5 Msun/Lsun with ~0.1-0.15 dex scatter
    (Meidt et al. 2014; Kettlety et al. 2018), and the W1 K-correction is
    mild and nearly color-independent (SED slope alpha ~ -2, Rayleigh-Jeans):
        K_W1 = -2.5 (1+alpha) log10(1+z) = +2.5 log10(1+z).
    This avoids the catastrophic failure of optical color-based M*/L at
    z >~ 0.7, where a flat-SED K-correction overestimates red-galaxy
    luminosities by ~2 mag. Galaxies without a W1 detection fall back to
    Taylor2011 optical colors.
    """

    MSUN_W1_AB = 5.92   # absolute AB magnitude of the Sun in W1
    ML_W1 = 0.5         # Msun / Lsun,W1
    ALPHA = -2.0        # Fnu ~ nu^alpha near rest 3.4 um

    def __init__(self, cosmo):
        self.cosmo = cosmo
        self._taylor = Taylor2011(cosmo)

    def logmstar(self, mags, z):
        z = np.asarray(z, dtype=float)
        w1 = np.asarray(mags.get("w1", np.full_like(z, np.nan)), dtype=float)
        dm = self.cosmo.distmod(np.clip(z, 1e-3, None)).value
        kcorr = -2.5 * (1.0 + self.ALPHA) * np.log10(1.0 + z)
        m_abs = w1 - dm - kcorr
        logm_w1 = np.log10(self.ML_W1) - 0.4 * (m_abs - self.MSUN_W1_AB)
        logm = np.where(np.isfinite(logm_w1), logm_w1,
                        self._taylor.logmstar(mags, z))
        return np.clip(logm, LOGM_MIN, LOGM_MAX)


class Nir1um(StellarMassEstimator):
    """Rest-frame 1 um stellar mass via z-band <-> W1 flux interpolation.

    Rest-frame ~1 um light is a low-scatter stellar-mass tracer
    (M*/L_1um ~ 0.6 Msun/Lsun, ~0.1-0.2 dex across SED types). For any
    0 < z < 2.4, the observed wavelength of rest 1 um, lambda_t = (1+z) um,
    falls between the observed z band (0.92 um) and WISE W1 (3.4 um), so each
    galaxy's own two-point power-law slope provides its K-correction --
    data-driven, no template assumption. This is what fixes the catastrophic
    failure of constant-slope K-corrections at z >~ 0.7 (rest-1.6-um-bump
    territory), which inflated masses by up to ~1.6 dex.

    M_AB(rest 1um) = m_AB(lambda_t) - DM(z) + 2.5 log10(1+z)   [Hogg 2002,
    band-shift K-correction], then log M* = log(M*/L) - 0.4 (M - M_sun,1um).
    Galaxies without W1 fall back to Taylor2011 optical colors.
    """

    LAM_Z, LAM_W1 = 0.92, 3.4       # microns, observed
    MSUN_1UM_AB = 4.52              # absolute AB mag of the Sun at 1 um
    ML_1UM = 0.6                    # Msun / Lsun at rest 1 um

    def __init__(self, cosmo):
        self.cosmo = cosmo
        self._taylor = Taylor2011(cosmo)

    def logmstar(self, mags, z):
        z = np.asarray(z, dtype=float)
        m_z = np.asarray(mags["z"], dtype=float)
        m_w1 = np.asarray(mags.get("w1", np.full_like(m_z, np.nan)),
                          dtype=float)
        lam_t = np.clip(1.0 * (1.0 + z), self.LAM_Z, self.LAM_W1)
        # log-log interpolation of AB mags in wavelength (AB mag is linear in
        # log Fnu, so linear interpolation in log(lambda) is a power law)
        frac = (np.log(lam_t / self.LAM_Z)
                / np.log(self.LAM_W1 / self.LAM_Z))
        m_t = m_z + frac * (m_w1 - m_z)
        dm = self.cosmo.distmod(np.clip(z, 1e-3, None)).value
        m_abs = m_t - dm + 2.5 * np.log10(1.0 + z)
        logm_nir = (np.log10(self.ML_1UM)
                    - 0.4 * (m_abs - self.MSUN_1UM_AB))
        logm = np.where(np.isfinite(logm_nir), logm_nir,
                        self._taylor.logmstar(mags, z))
        return np.clip(logm, LOGM_MIN, LOGM_MAX)


class Nir1umFSF(Nir1um):
    """Nir1um recalibrated to the DESI DR1 FastSpecFit mass scale.

    The constant M*/L_1um = 0.6 underestimates massive red galaxies by
    0.2-0.25 dex (old stellar populations carry higher rest-1um M/L) -- a
    deficit diagnosed by the galaxy-galaxy lensing closure test
    (scripts/delta_sigma_closure.py) and pinned by a per-galaxy comparison
    with the public FastSpecFit VAC (scripts/mstar_vs_fastspecfit.py).
    This estimator adds the empirical median correction
    Delta(logM*_raw, z) built by scripts/build_mstar_recal.py from
    ~500k DESI DR1 BGS+LRG galaxies, bilinearly interpolated and clamped
    to the calibration grid (faint/low-mass galaxies therefore extrapolate
    the nearest calibrated cell; they carry little lensing weight).
    """

    def __init__(self, cosmo, table_path=None):
        super().__init__(cosmo)
        import json
        from pathlib import Path
        p = (Path(table_path) if table_path
             else Path(__file__).parent / "data" / "nir1um_fsf_recal.json")
        t = json.loads(p.read_text())
        self._c0 = float(t["c0"]); self._m0 = float(t["m0"])
        self._w = float(t["w"])
        self._zn = np.asarray(t["z_nodes"], dtype=float)
        self._an = np.asarray(t["a_nodes"], dtype=float)

    def _delta(self, m, z):
        a = np.interp(np.clip(z, self._zn[0], self._zn[-1]),
                      self._zn, self._an)
        return self._c0 + a / (1.0 + np.exp(-(m - self._m0) / self._w))

    def logmstar(self, mags, z):
        raw = super().logmstar(mags, z)
        z = np.asarray(z, dtype=float)
        # Delta is calibrated against the corrected (FSF-scale) mass, so
        # apply it by fixed-point iteration starting from the raw estimate
        m = raw
        for _ in range(3):
            m = raw + self._delta(m, z)
        return np.clip(m, LOGM_MIN, LOGM_MAX)


def make_estimator(name: str, cosmo) -> StellarMassEstimator:
    if name == "taylor2011":
        return Taylor2011(cosmo)
    if name == "w1taylor":
        return W1Taylor(cosmo)
    if name == "nir1um":
        return Nir1um(cosmo)
    if name == "nir1um_fsf":
        return Nir1umFSF(cosmo)
    raise ValueError(f"Unknown mstar_method: {name}")
