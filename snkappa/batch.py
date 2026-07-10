"""Survey-mode (batch) kappa pipeline: many sightlines against one regional
catalog, with per-source-redshift-bin Sigma_crit tables.

Shared by scripts/des_full.py and its variant runs (moved out of the script,
TODO 3.9, so the batch path uses the package's guards instead of drifting as
a parallel implementation).

Two components:

- BatchEngine: per-galaxy halo tables built ONCE per region (z_src
  independent); only Sigma_crit and the photo-z truncation are recomputed per
  source-redshift bin. Supports the host-environment excision variant
  (galaxies with redshift consistent with the source plane removed) and the
  W1-required variant.

- ClusterField: deterministic cluster convergence. Each cluster's surface
  density is marginalized ONCE over (a) a centered lognormal richness-mass
  scatter (Gauss-Hermite quadrature with E[M] = M_catalog -- the catalog
  mass is treated as unbiased, so no silent Jensen boost; the old MC drew
  10^(0.25 N) with E[M] = 1.18 M_catalog) and (b) Rayleigh miscentering with
  a fixed PHYSICAL scale sigma_mis = miscenter_frac_r500 * r500 (the old
  fixed 30 arcsec conflated very different physical offsets across cluster
  redshift). The result is a per-cluster radial profile, interpolated at
  evaluation time: no per-sightline MC jitter attenuating the regression.
"""

from __future__ import annotations

import numpy as np

from . import photoz
from .kappa import angular_sep_arcsec, sigma_crit_msun_mpc2

ARCSEC_PER_RAD = 206264.806
R500_OVER_R200 = 0.66   # NFW c200 ~ 5


class BatchEngine:
    """Halo tables built once; Sigma_crit / photo-z truncation set per z_src.

    Parameters
    ----------
    cfg : Config (halo model, cosmology, apertures)
    df : regional catalog (clean_and_merge output, cluster members removed)
    hm : HaloModel built for the maximum source redshift
    est : StellarMassEstimator
    zc_grid : photo-z marginalization grid
    r_in_arcsec : inner exclusion radius
    require_w1 : drop galaxies without a W1 detection (variant 1.5) instead
        of letting them fall back to the optical-color estimator
    """

    def __init__(self, cfg, df, hm, est, zc_grid, r_in_arcsec=3.0,
                 require_w1=False):
        self.hm = hm
        self.zc = np.asarray(zc_grid, dtype=float)
        self.r_in = float(r_in_arcsec)
        if require_w1 and "mag_w1" in df:
            df = df[np.isfinite(df["mag_w1"])].reset_index(drop=True)
        self.n_gal = len(df)
        self.all_ra = df.ra.to_numpy()
        self.all_dec = df.dec.to_numpy()

        spec = df.z_spec.notna().to_numpy()
        s = df[spec]
        p = df[~spec]
        self.s_ra = s.ra.to_numpy(); self.s_dec = s.dec.to_numpy()
        self.s_z = s.z_spec.to_numpy(dtype=float)
        ib = hm.zbin_index(self.s_z)
        mags = {b: s[f"mag_{b}"].to_numpy(float)
                for b in ("g", "r", "i", "z", "w1") if f"mag_{b}" in s}
        logms = est.logmstar(mags, hm.zbins[ib])
        rhos, rs, tau = hm.halo_params(logms, ib)
        self.s_ib = ib
        self.s_amp0 = rhos * rs
        self.s_ths = rs / hm.da[ib] * ARCSEC_PER_RAD
        self.s_tau = tau

        self.p_ra = p.ra.to_numpy(); self.p_dec = p.dec.to_numpy()
        n = len(p)
        zc = self.zc
        izc = hm.zbin_index(zc)
        self.p_amp0 = np.empty((n, zc.size), np.float32)
        self.p_ths = np.empty((n, zc.size), np.float32)
        self.p_tau = np.empty((n, zc.size), np.float32)
        magp = {b: p[f"mag_{b}"].to_numpy(float)
                for b in ("g", "r", "i", "z", "w1") if f"mag_{b}" in p}
        for j, zcj in enumerate(zc):
            lm = est.logmstar(magp, np.full(n, zcj))
            rhos, rs, tau = hm.halo_params(lm, np.full(n, izc[j]))
            self.p_amp0[:, j] = rhos * rs
            self.p_ths[:, j] = rs / hm.da[izc[j]] * ARCSEC_PER_RAD
            self.p_tau[:, j] = tau
        mu, slo, shi = photoz.split_normal_params(
            p.zp_med.to_numpy(float), p.zp_l68.to_numpy(float),
            p.zp_u68.to_numpy(float), p.zp_std.to_numpy(float))
        self.p_mu, self.p_slo, self.p_shi = mu, slo, shi
        dzc = np.gradient(zc)
        self.p_pdfw = (photoz.grid_pdf(zc, mu, slo, shi)
                       * dzc[None, :]).astype(np.float32)
        self.izc = izc

    def table_bytes(self):
        return self.p_amp0.nbytes * 3 + self.p_pdfw.nbytes

    def set_zsrc(self, cosmo, z_src, excise_frac=None):
        """Prepare Sigma_crit-weighted amplitudes for one source plane.

        excise_frac: host-environment robustness variant (TODO 1.2a) --
        remove foreground weight within |z - z_src| < excise_frac*(1+z_src)
        (spec-z galaxies masked; photo-z grid columns zeroed). Applied
        identically to SN and random sightlines, so the zero point stays
        consistent.
        """
        keep_s = self.s_z < z_src - 0.01
        keep_c = self.zc < z_src - 0.01
        if excise_frac:
            win = excise_frac * (1.0 + z_src)
            keep_s &= np.abs(self.s_z - z_src) >= win
            keep_c &= np.abs(self.zc - z_src) >= win
        sig_fine = sigma_crit_msun_mpc2(cosmo, self.hm.zbins, z_src)
        self.sA = np.where(keep_s, self.s_amp0 / sig_fine[self.s_ib], 0.0)
        sig_zc = sigma_crit_msun_mpc2(cosmo, self.zc, z_src)
        w = self.p_pdfw * keep_c[None, :]
        with np.errstate(divide="ignore"):
            self.pWA = (w * (self.p_amp0 / sig_zc[None, :])).astype(np.float32)

    @staticmethod
    def _band(dec, dec0, r_out):
        """Cheap declination-band prefilter before the haversine."""
        return np.abs(dec - dec0) < r_out / 3600.0

    def kappa_gal(self, ra0, dec0, r_out):
        """Summed galaxy-halo convergence at (ra0, dec0)."""
        b = self._band(self.s_dec, dec0, r_out)
        th = angular_sep_arcsec(ra0, dec0, self.s_ra[b], self.s_dec[b])
        m = (th > self.r_in) & (th < r_out) & (self.sA[b] > 0)
        k = 0.0
        if m.any():
            sA = self.sA[b][m]
            k = float(np.sum(sA * self.hm.sigma_dimless(
                th[m] / self.s_ths[b][m], self.s_tau[b][m])))
        b = self._band(self.p_dec, dec0, r_out)
        th = angular_sep_arcsec(ra0, dec0, self.p_ra[b], self.p_dec[b])
        m = (th > self.r_in) & (th < r_out)
        if m.any():
            ths = self.p_ths[b][m]
            x = th[m][:, None] / ths
            sig = self.hm.sigma_dimless(
                x.ravel(), self.p_tau[b][m].ravel()).reshape(x.shape)
            k += float(np.sum(self.pWA[b][m] * sig))
        return k

    def counts(self, ra0, dec0, r_out):
        """Catalog galaxies in the aperture (masking / edge diagnostic)."""
        b = self._band(self.all_dec, dec0, r_out)
        th = angular_sep_arcsec(ra0, dec0, self.all_ra[b], self.all_dec[b])
        return int(np.count_nonzero((th > self.r_in) & (th < r_out)))


class ClusterField:
    """Deterministic miscentering- and mass-marginalized cluster kappa.

    Per cluster, precomputes E[Sigma](r) on a log radial grid, marginalized
    over a CENTERED lognormal mass scatter (E[M] = M_catalog; Gauss-Hermite,
    n_mass nodes) and Rayleigh miscentering with sigma_mis =
    miscenter_frac_r500 * r500 (quantile quadrature x azimuthal average).
    Evaluation is then a table interpolation: fully deterministic and cheap.
    """

    R_MAX_ARCSEC = 1800.0   # clusters beyond 0.5 deg contribute negligibly

    def __init__(self, cl, hm, mass_scatter_dex=0.25, miscenter_frac_r500=0.2,
                 conc=5.0, n_grid=44, n_mass=5, n_mis=8, n_phi=16):
        self.hm = hm
        self.z = cl.z.to_numpy(dtype=float)
        self.ra = cl.ra.to_numpy(dtype=float)
        self.dec = cl.dec.to_numpy(dtype=float)
        m0 = cl.m200.to_numpy(dtype=float)
        n_cl = self.z.size
        self.ib = hm.zbin_index(self.z)

        # centered lognormal mass nodes: M_j = M0 10^(s sqrt(2) t_j) / E[10^(sN)]
        t, wt = np.polynomial.hermite.hermgauss(n_mass)
        wt = wt / np.sqrt(np.pi)
        s = mass_scatter_dex
        mass_fac = 10.0 ** (s * np.sqrt(2.0) * t) \
            * np.exp(-0.5 * (s * np.log(10.0)) ** 2)
        self.mass_nodes = mass_fac
        self.mass_wts = wt

        # Rayleigh quantile nodes (equal weight) for the miscentering radius
        u = (np.arange(n_mis) + 0.5) / n_mis
        ray = np.sqrt(-2.0 * np.log(1.0 - u))       # in units of sigma_mis
        phi = (np.arange(n_phi) + 0.5) / n_phi * 2.0 * np.pi

        self.loggrid = np.linspace(0.0, np.log10(2.5e3), n_grid)  # arcsec
        grid = 10.0 ** self.loggrid
        self.profiles = np.empty((n_cl, n_grid))   # E[Sigma] [Msun/Mpc^2]

        rhoc = hm.rhoc[self.ib]
        da = hm.da[self.ib]
        for i in range(n_cl):
            m_j = m0[i] * mass_fac                              # [n_mass]
            r200 = (3.0 * m_j / (4.0 * np.pi * 200.0 * rhoc[i])) ** (1 / 3)
            rs = r200 / conc                                    # [n_mass] Mpc
            rhos = (200.0 / 3.0) * rhoc[i] * conc ** 3 \
                / (np.log(1.0 + conc) - conc / (1.0 + conc))
            sig_mis = miscenter_frac_r500 * R500_OVER_R200 * r200.mean()
            r_mis = sig_mis * ray                               # [n_mis] Mpc
            r_g = grid / ARCSEC_PER_RAD * da[i]                 # [n_grid] Mpc
            # offset radii [n_grid, n_mis, n_phi]
            d = np.sqrt(r_g[:, None, None] ** 2 + r_mis[None, :, None] ** 2
                        + 2.0 * r_g[:, None, None] * r_mis[None, :, None]
                        * np.cos(phi)[None, None, :])
            x = np.clip(d[None] / rs[:, None, None, None], 1e-3, None)
            sig = hm.sigma_dimless(
                x.ravel(), np.full(x.size, conc)).reshape(x.shape)
            prof = np.einsum("j,jgmp->g", wt * rhos * rs, sig) \
                / (ray.size * phi.size)
            self.profiles[i] = prof
        self.logprof = np.log10(np.clip(self.profiles, 1e-30, None))

    def set_zsrc(self, cosmo, z_src):
        self.sigcr = sigma_crit_msun_mpc2(
            cosmo, self.hm.zbins[self.ib], z_src)
        self.fg = self.z < z_src - 0.02

    def kappa_sum(self, ra0, dec0):
        """Summed cluster kappa at a sightline (call set_zsrc first)."""
        dx = (self.ra - ra0) * np.cos(np.radians(dec0)) * 3600.0
        dy = (self.dec - dec0) * 3600.0
        sep = np.hypot(dx, dy)
        near = (sep < self.R_MAX_ARCSEC) & self.fg
        if not near.any():
            return 0.0
        ls = np.log10(np.clip(sep[near], 1.0, None))
        idx = np.clip(np.searchsorted(self.loggrid, ls) - 1, 0,
                      self.loggrid.size - 2)
        f = (ls - self.loggrid[idx]) / (self.loggrid[idx + 1]
                                        - self.loggrid[idx])
        lp = self.logprof[near, idx] * (1 - f) + self.logprof[near, idx + 1] * f
        return float(np.sum(10.0 ** lp / self.sigcr[near]))
