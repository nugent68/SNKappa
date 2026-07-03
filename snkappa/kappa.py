"""Convergence engine: per-galaxy kappa_i for arbitrary sightlines.

kappa_i = Sigma_halo(b_i) / Sigma_crit(z_i, z_src), b_i = theta_i * D_A(z_i).
Spec-z galaxies are evaluated at their redshift; photo-z galaxies are
marginalized over p(z) on a coarse subset of the fine z grid, with the
z < z_src lensing condition arising naturally (weights beyond z_src are never
included, so PDF mass at z >= z_src down-weights the galaxy).

Single-plane approximation throughout (each halo treated as an independent
thin lens; no multi-plane coupling).
"""

from __future__ import annotations

import numpy as np
from astropy import constants as const
from astropy import units as u

from . import photoz


def d_a_z1z2(cosmo, z1, z2):
    """Angular diameter distance between two redshifts (astropy 6/7 compat)."""
    try:
        return cosmo.angular_diameter_distance(z1, z2)
    except TypeError:
        return cosmo.angular_diameter_distance_z1z2(z1, z2)


def sigma_crit_msun_mpc2(cosmo, zl, zs):
    """Sigma_crit(z_l, z_s) in Msun / physical Mpc^2, vectorized over zl."""
    zl = np.atleast_1d(np.asarray(zl, dtype=float))
    prefac = (const.c**2 / (4.0 * np.pi * const.G)).to(u.Msun / u.Mpc).value
    d_s = cosmo.angular_diameter_distance(zs).value
    d_l = cosmo.angular_diameter_distance(zl).value
    d_ls = d_a_z1z2(cosmo, zl, zs).value
    with np.errstate(divide="ignore", invalid="ignore"):
        scr = prefac * d_s / (d_l * d_ls)
    scr[(d_ls <= 0) | (d_l <= 0)] = np.inf
    return scr


def angular_sep_arcsec(ra0, dec0, ra, dec):
    """Great-circle separation in arcsec (haversine, vectorized)."""
    ra0r, dec0r = np.radians(ra0), np.radians(dec0)
    rar, decr = np.radians(np.asarray(ra)), np.radians(np.asarray(dec))
    sd = np.sin(0.5 * (decr - dec0r))
    sr = np.sin(0.5 * (rar - ra0r))
    h = sd**2 + np.cos(dec0r) * np.cos(decr) * sr**2
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600.0


class KappaEngine:
    """Precomputes fiducial halo tables for the regional catalog; evaluates
    per-galaxy kappa for any LOS center (SN or random)."""

    N_ZCOARSE = 30  # photo-z marginalization grid size

    def __init__(self, cfg, cosmo, halo_model, stellar_est, df,
                 logms_override=None, logms_err=None, calib_offset=0.0,
                 mstar_scatter=None):
        """logms_override/logms_err: per-df-row arrays (NaN = use estimator);
        calib_offset is SUBTRACTED from the base estimator (measured
        cheap-minus-FB bias); mstar_scatter overrides the config M* scatter
        for non-overridden galaxies in the Monte Carlo."""
        self.cfg = cfg
        self.cosmo = cosmo
        self.hm = halo_model
        self.stellar = stellar_est
        self.df = df.reset_index(drop=True)
        n = len(self.df)
        self.logms_override = (np.full(n, np.nan) if logms_override is None
                               else np.asarray(logms_override, dtype=float))
        self.logms_err = (np.full(n, np.nan) if logms_err is None
                          else np.asarray(logms_err, dtype=float))
        self.calib_offset = float(calib_offset)
        self.mstar_scatter = (cfg.halo_model.mstar_scatter_dex
                              if mstar_scatter is None else float(mstar_scatter))

        z_src = cfg.source.z_src
        self.sigcr = sigma_crit_msun_mpc2(cosmo, self.hm.zbins, z_src)
        # lensing efficiency D_l * D_ls / D_s on the fine grid (for zeta weights)
        d_ls = d_a_z1z2(cosmo, self.hm.zbins, z_src).value
        d_s = cosmo.angular_diameter_distance(z_src).value
        self.efficiency = np.clip(self.hm.da * d_ls / d_s, 0, None)
        self.efficiency /= self.efficiency.max()

        self.is_spec = self.df["z_spec"].notna().to_numpy()
        self._prep_spec()
        self._prep_phot()

    def logmstar_at(self, gal_idx, mags, z):
        """Stellar mass for df rows gal_idx at redshift(s) z: FB override
        where available, else calibrated base estimator."""
        base = self.stellar.logmstar(mags, z) - self.calib_offset
        ov = self.logms_override[gal_idx]
        return np.where(np.isfinite(ov), ov, base)

    # -- fiducial tables ----------------------------------------------------

    def _prep_spec(self):
        s = self.df[self.is_spec]
        self.spec_idx = np.flatnonzero(self.is_spec)
        z = s["z_spec"].to_numpy(dtype=float)
        ibin = self.hm.zbin_index(z)
        mags = {b: s[f"mag_{b}"].to_numpy(dtype=float)
                for b in ("g", "r", "i", "z", "w1") if f"mag_{b}" in s}
        logms = self.logmstar_at(self.spec_idx, mags, self.hm.zbins[ibin])
        rhos, rs, tau = self.hm.halo_params(logms, ibin)
        self.spec_ibin = ibin
        self.spec_logms = logms
        self.spec_amp = (rhos * rs / self.sigcr[ibin]).astype(np.float64)
        self.spec_theta_s = (rs / self.hm.da[ibin]) * 206264.806  # arcsec
        self.spec_tau = tau

    def _prep_phot(self):
        p = self.df[~self.is_spec]
        self.phot_idx = np.flatnonzero(~self.is_spec)
        stride = max(1, self.hm.zbins.size // self.N_ZCOARSE)
        self.zc_ibins = np.arange(0, self.hm.zbins.size, stride)
        zc = self.hm.zbins[self.zc_ibins]

        mu, slo, shi = photoz.split_normal_params(
            p["zp_med"].to_numpy(dtype=float),
            p["zp_l68"].to_numpy(dtype=float),
            p["zp_u68"].to_numpy(dtype=float),
            p["zp_std"].to_numpy(dtype=float))
        self.phot_mu, self.phot_slo, self.phot_shi = mu, slo, shi
        # weights [n_phot, n_zc]; grid spacing accounts for the stride
        w = photoz.grid_pdf(zc, mu, slo, shi)
        dz = np.gradient(zc)
        self.phot_w = (w * dz[None, :]).astype(np.float32)

        n_p, n_z = len(p), zc.size
        self.phot_amp = np.empty((n_p, n_z), dtype=np.float32)
        self.phot_theta_s = np.empty((n_p, n_z), dtype=np.float32)
        self.phot_tau = np.empty((n_p, n_z), dtype=np.float32)
        mags = {b: p[f"mag_{b}"].to_numpy(dtype=float)
                for b in ("g", "r", "i", "z", "w1") if f"mag_{b}" in p}
        for j, ib in enumerate(self.zc_ibins):
            zj = self.hm.zbins[ib]
            logms = self.logmstar_at(self.phot_idx, mags, np.full(n_p, zj))
            rhos, rs, tau = self.hm.halo_params(logms, np.full(n_p, ib))
            self.phot_amp[:, j] = rhos * rs / self.sigcr[ib]
            self.phot_theta_s[:, j] = rs / self.hm.da[ib] * 206264.806
            self.phot_tau[:, j] = tau

    # -- evaluation ---------------------------------------------------------

    def kappa_los(self, ra0, dec0, r_inner_arcsec, r_outer_arcsec,
                  exclude_mask=None):
        """Per-galaxy fiducial kappa for a sightline centered at (ra0, dec0).

        Galaxies with r_inner < theta < r_outer contribute; `exclude_mask`
        (boolean over the full df) removes e.g. deflector/group members.
        Returns (indices into df, theta_arcsec, kappa_i).
        """
        theta = angular_sep_arcsec(ra0, dec0, self.df["ra"].to_numpy(),
                                   self.df["dec"].to_numpy())
        sel = (theta > r_inner_arcsec) & (theta < r_outer_arcsec)
        if exclude_mask is not None:
            sel &= ~exclude_mask
        kappa = np.zeros(len(self.df))

        # spec-z galaxies
        m = sel[self.spec_idx]
        if m.any():
            gi = self.spec_idx[m]
            x = theta[gi] / self.spec_theta_s[m]
            kappa[gi] = self.spec_amp[m] * self.hm.sigma_dimless(
                x, self.spec_tau[m])

        # photo-z galaxies (marginalized)
        m = sel[self.phot_idx]
        if m.any():
            gi = self.phot_idx[m]
            x = theta[gi][:, None] / self.phot_theta_s[m]
            sig = self.hm.sigma_dimless(x.ravel(),
                                        self.phot_tau[m].ravel()).reshape(x.shape)
            kappa[gi] = np.sum(self.phot_w[m] * self.phot_amp[m] * sig, axis=1)

        idx = np.flatnonzero(sel)
        return idx, theta[idx], kappa[idx]

    def weighted_counts(self, ra0, dec0, r_inner_arcsec, r_outer_arcsec,
                        exclude_mask=None):
        """Number counts with the H0LiCOW-style weights: (N, N_1/r, N_eff)."""
        theta = angular_sep_arcsec(ra0, dec0, self.df["ra"].to_numpy(),
                                   self.df["dec"].to_numpy())
        sel = (theta > r_inner_arcsec) & (theta < r_outer_arcsec)
        if exclude_mask is not None:
            sel &= ~exclude_mask

        # per-galaxy lensing-efficiency weight (photo-z: PDF-weighted)
        eff = np.zeros(len(self.df))
        eff[self.spec_idx] = self.efficiency[self.spec_ibin]
        eff[self.phot_idx] = np.sum(
            self.phot_w * self.efficiency[self.zc_ibins][None, :], axis=1)

        n = float(np.count_nonzero(sel))
        inv_r = np.sum(np.clip(r_inner_arcsec, 1.0, None) / theta[sel])
        n_eff = float(np.sum(eff[sel]))
        return n, float(inv_r), n_eff
