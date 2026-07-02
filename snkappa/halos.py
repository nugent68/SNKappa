"""Galaxy -> halo mapping and halo surface-density profiles.

Chain: log M* --(SMHM inverse: Behroozi+2013 default / Moster+2013)--> M_200c
       --(c(M,z): Diemer & Joyce 2019 via colossus)--> (rho_s, r_s, tau=c)
       --> truncated-NFW Sigma(R) (BMO 2009, n=1 truncation at r_200c).

Unit conventions: masses in Msun, lengths in physical Mpc, densities in
Msun/Mpc^3, surface densities in Msun/Mpc^2. colossus is used only for the
concentration model (fed Msun/h) to avoid little-h traps elsewhere.

Redshifts are quantized to a fine grid (dz=0.01) for all halo-parameter
computations; the quantization error is far below photo-z uncertainty and the
SMHM/c scatter.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator, interp1d

# ---------------------------------------------------------------------------
# SMHM relations (forward: logM* as a function of logMh at redshift z)
# ---------------------------------------------------------------------------


def behroozi13_logmstar(logmh, z):
    """Behroozi, Wechsler & Conroy 2013 (ApJ 770, 57) intrinsic SMHM relation."""
    a = 1.0 / (1.0 + z)
    nu = np.exp(-4.0 * a * a)
    log_eps = -1.777 + (-0.006 * (a - 1.0)) * nu - 0.119 * (a - 1.0)
    log_m1 = 11.514 + (-1.793 * (a - 1.0) + (-0.251) * z) * nu
    alpha = -1.412 + (0.731 * (a - 1.0)) * nu
    delta = 3.508 + (2.608 * (a - 1.0) + (-0.043) * z) * nu
    gamma = 0.316 + (1.319 * (a - 1.0) + 0.279 * z) * nu

    def f(x):
        return (-np.log10(10.0 ** (alpha * x) + 1.0)
                + delta * np.log10(1.0 + np.exp(x)) ** gamma
                / (1.0 + np.exp(10.0 ** (-x))))

    x = np.asarray(logmh) - log_m1
    return log_eps + log_m1 + f(x) - f(0.0)


def moster13_logmstar(logmh, z):
    """Moster, Naab & White 2013 (MNRAS 428, 3121) SMHM relation."""
    zf = z / (1.0 + z)
    log_m1 = 11.590 + 1.195 * zf
    n = 0.0351 - 0.0247 * zf
    beta = 1.376 - 0.826 * zf
    gamma = 0.608 + 0.329 * zf
    mh_m1 = 10.0 ** (np.asarray(logmh) - log_m1)
    ratio = 2.0 * n / (mh_m1 ** (-beta) + mh_m1 ** gamma)
    return np.asarray(logmh) + np.log10(ratio)


_SMHM_FORWARD = {"behroozi13": behroozi13_logmstar, "moster13": moster13_logmstar}


# ---------------------------------------------------------------------------
# Dimensionless surface-density profiles: sigma_tilde(x) = Sigma / (rho_s r_s)
# ---------------------------------------------------------------------------


def nfw_sigma_dimless(x):
    """Analytic NFW Sigma/(rho_s r_s) (Wright & Brainerd 2000)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.full_like(x, 2.0 / 3.0)
    lo = x < 1.0 - 1e-8
    hi = x > 1.0 + 1e-8
    xl, xh = x[lo], x[hi]
    out[lo] = (2.0 / (xl**2 - 1.0)) * (
        1.0 - 2.0 / np.sqrt(1.0 - xl**2)
        * np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl))))
    out[hi] = (2.0 / (xh**2 - 1.0)) * (
        1.0 - 2.0 / np.sqrt(xh**2 - 1.0)
        * np.arctan(np.sqrt((xh - 1.0) / (xh + 1.0))))
    return out


def bmo_sigma_direct(x, tau, n_quad=128):
    """Truncated-NFW Sigma/(rho_s r_s) by direct LOS integration.

    rho(r) = rho_s / (u (1+u)^2) * (tau^2/(tau^2+u^2)),  u = r/r_s  (BMO n=1).
    Sigma(x) = 2 x * int_0^thetamax rho_tilde(x cosh t) cosh t dt.
    Used to build the interpolation table and as the truth in unit tests.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    tau = float(tau)
    umax = np.maximum(1e5, 1e3 * tau)
    tmax = np.arccosh(np.maximum(umax / x, 1.0 + 1e-10))
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    t = 0.5 * tmax[:, None] * (nodes[None, :] + 1.0)
    w = 0.5 * tmax[:, None] * weights[None, :]
    uu = x[:, None] * np.cosh(t)
    rho = 1.0 / (uu * (1.0 + uu) ** 2) * (tau**2 / (tau**2 + uu**2))
    return 2.0 * x * np.sum(rho * np.cosh(t) * w, axis=1)


class BMOTable:
    """2D log-log interpolation table for the BMO truncated-NFW Sigma."""

    def __init__(self, n_x=512, n_tau=48):
        self.logx_grid = np.linspace(-4.0, 3.0, n_x)
        self.logtau_grid = np.linspace(np.log10(0.5), np.log10(300.0), n_tau)
        table = np.empty((n_tau, n_x))
        xg = 10.0 ** self.logx_grid
        for j, logtau in enumerate(self.logtau_grid):
            table[j] = np.log10(np.clip(bmo_sigma_direct(xg, 10.0 ** logtau),
                                        1e-300, None))
        self._interp = RegularGridInterpolator(
            (self.logtau_grid, self.logx_grid), table,
            bounds_error=False, fill_value=None)

    def sigma_dimless(self, x, tau):
        x = np.clip(np.asarray(x, dtype=float), 1e-4, 1e3)
        tau = np.clip(np.asarray(tau, dtype=float), 0.5, 300.0)
        pts = np.stack([np.log10(np.broadcast_to(tau, x.shape)),
                        np.log10(x)], axis=-1)
        return 10.0 ** self._interp(pts)


_BMO_TABLE: BMOTable | None = None


def bmo_table() -> BMOTable:
    global _BMO_TABLE
    if _BMO_TABLE is None:
        _BMO_TABLE = BMOTable()
    return _BMO_TABLE


# ---------------------------------------------------------------------------
# HaloModel: the full chain on a quantized redshift grid
# ---------------------------------------------------------------------------


class HaloModel:
    def __init__(self, halo_cfg, cosmo, z_src, dz=0.01):
        self.cfg = halo_cfg
        self.cosmo = cosmo
        self.z_src = float(z_src)
        self.zbins = np.arange(dz, self.z_src, dz)
        self.dz = dz

        self._setup_colossus()
        self._forward = _SMHM_FORWARD[halo_cfg.smhm]
        self._build_smhm_inverse()

        # per-bin critical density [Msun / physical Mpc^3]
        self.rhoc = self.cosmo.critical_density(self.zbins).to(
            u.Msun / u.Mpc**3).value
        # per-bin angular diameter distance [Mpc]
        self.da = self.cosmo.angular_diameter_distance(self.zbins).value

        self._profile = halo_cfg.profile
        if self._profile not in ("bmo", "nfw"):
            raise ValueError(f"Unknown profile: {self._profile}")

    def _setup_colossus(self):
        from colossus.cosmology import cosmology as ccosmo
        params = {
            "flat": True,
            "H0": self.cosmo.H0.value,
            "Om0": self.cosmo.Om0,
            "Ob0": 0.048,
            "sigma8": 0.81,
            "ns": 0.96,
        }
        ccosmo.setCosmology("snkappa", **params)
        self._h = self.cosmo.H0.value / 100.0

    def _build_smhm_inverse(self):
        """Monotonic inverse interpolators logM* -> logMh per z bin."""
        logmh_grid = np.linspace(9.5, 15.8, 160)
        self._inv = []
        for z in self.zbins:
            logms = self._forward(logmh_grid, z)
            # enforce strict monotonicity for interpolation
            logms = np.maximum.accumulate(logms)
            logms += np.arange(logms.size) * 1e-9
            self._inv.append(interp1d(
                logms, logmh_grid, bounds_error=False,
                fill_value=(logmh_grid[0], logmh_grid[-1])))

    def zbin_index(self, z):
        """Nearest fine-grid bin index; z outside [0, z_src) clipped/flagged."""
        idx = np.rint((np.asarray(z, dtype=float) - self.zbins[0])
                      / self.dz).astype(int)
        return np.clip(idx, 0, self.zbins.size - 1)

    def halo_params(self, logmstar, ibin, dlogm=None, dlogc=None):
        """(rho_s [Msun/Mpc^3], r_s [Mpc], tau) for galaxies at z-bin ibin.

        dlogm: additive SMHM-scatter draw in dex (same shape as logmstar);
        dlogc: additive lognormal concentration-scatter draw in dex.
        Vectorized; groups by unique z bin internally (colossus needs scalar z).
        """
        from colossus.halo import concentration

        logmstar = np.asarray(logmstar, dtype=float)
        ibin = np.asarray(ibin, dtype=int)
        logmh = np.empty_like(logmstar)
        conc = np.empty_like(logmstar)

        logmh_max = getattr(self.cfg, "logmh_max", 13.8)
        for b in np.unique(ibin):
            m = ibin == b
            lmh = self._inv[b](logmstar[m])
            if dlogm is not None:
                lmh = lmh + np.asarray(dlogm)[m]
            logmh[m] = np.minimum(lmh, logmh_max)
            c = concentration.concentration(
                10.0 ** lmh * self._h, "200c", float(self.zbins[b]),
                model=self.cfg.cmodel)
            conc[m] = c
        if dlogc is not None:
            conc = conc * 10.0 ** np.asarray(dlogc)
        conc = np.clip(conc, 1.0, 40.0)

        m200 = 10.0 ** logmh
        rhoc = self.rhoc[ibin]
        r200 = (3.0 * m200 / (4.0 * np.pi * 200.0 * rhoc)) ** (1.0 / 3.0)
        rs = r200 / conc
        mu_c = np.log(1.0 + conc) - conc / (1.0 + conc)
        rhos = (200.0 / 3.0) * rhoc * conc**3 / mu_c
        tau = conc  # truncation at r_200c
        return rhos, rs, tau

    def sigma_dimless(self, x, tau):
        if self._profile == "nfw":
            return nfw_sigma_dimless(x)
        return bmo_table().sigma_dimless(x, tau)
