"""Cluster-halo tier: intracluster mass beyond the sum of member galaxies.

Galaxy-sum halo models undercount clusters by construction: the common halo,
ICM, and stripped material belong to no galaxy. This module injects detected
clusters as single NFW halos REGION-WIDE (so random sightlines see them too
and the mass-sheet zero point stays consistent), removes their member
galaxies' halos (replacement, not addition), and propagates richness-mass
scatter and miscentering through the Monte Carlo.

Host-cluster subtlety: a cluster whose catalog center falls inside the
deflector exclusion radius hosts the lens itself. Its convergence at the
sight line is largely degenerate with the strong-lens model (its local
gradient is absorbed into theta_E and the external shear; its uniform part
IS the mass-sheet degeneracy). It is therefore reported as a SEPARATE
conditional term -- never folded into the headline kappa_ext -- and its
members are still replaced so the galaxy sum is not double-counted.

Default catalog: Wen & Han 2024 (ApJS 272, 39; DESI Legacy Surveys clusters)
fetched from VizieR TAP; manual entries may be added via the config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.stats import biweight_scale

from .kappa import angular_sep_arcsec, sigma_crit_msun_mpc2

C_KMS = 299792.458
WENHAN_TABLE = 'J/ApJS/272/39/table2'


def fetch_clusters(cfg, tap_factory) -> pd.DataFrame:
    """Cluster table over the analysis region: columns
    [name, ra, dec, z, m200, source]. tap_factory(url) -> TapClient."""
    from .catalog import region_radius_deg

    ccfg = cfg.clusters
    frames = []
    if ccfg.source in ("wenhan2024", "both"):
        tap = tap_factory(ccfg.vizier_tap)
        radius = region_radius_deg(cfg)
        q = (f'SELECT Name, RAJ2000, DEJ2000, zCl, M500 '
             f'FROM "{WENHAN_TABLE}" WHERE 1=CONTAINS('
             f"POINT('ICRS',RAJ2000,DEJ2000), CIRCLE('ICRS',"
             f"{cfg.source.ra_src},{cfg.source.dec_src},{radius}))")
        wh = tap.query(q, label="clusters:wenhan2024")
        wh = wh.rename(columns={"Name": "name", "RAJ2000": "ra",
                                "DEJ2000": "dec", "zCl": "z"})
        wh["m200"] = wh["M500"].astype(float) * 1e14 * ccfg.m200_from_m500
        wh["source"] = "wenhan2024"
        frames.append(wh[["name", "ra", "dec", "z", "m200", "source"]])
    for entry in ccfg.manual:
        m200 = entry.get("m200")
        if m200 is None and "m500" in entry:
            m200 = entry["m500"] * ccfg.m200_from_m500
        if m200 is None and "sigma_v" in entry:
            m200 = m200_from_sigma(entry["sigma_v"], entry["z"], cfg.cosmo)
        frames.append(pd.DataFrame([{
            "name": entry.get("name", "manual"), "ra": entry["ra"],
            "dec": entry["dec"], "z": entry["z"], "m200": m200,
            "source": "manual"}]))
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["m200"]) & (df["m200"] > 0)
            & (df["z"] > 0) & (df["z"] < cfg.source.z_src)]
    return df.reset_index(drop=True)


def split_host(cfg, clusters: pd.DataFrame):
    """(host_clusters, field_clusters): host = centered within the deflector
    exclusion radius, i.e. the structure the strong-lens model lives in."""
    sep = angular_sep_arcsec(cfg.deflector.ra_lens, cfg.deflector.dec_lens,
                             clusters["ra"].to_numpy(),
                             clusters["dec"].to_numpy())
    is_host = sep < max(cfg.deflector.r_exclude_arcsec, 30.0)
    return clusters[is_host].copy(), clusters[~is_host].copy()


# ---------------------------------------------------------------------------
# NFW pieces for cluster-scale halos (reuses the profile machinery)
# ---------------------------------------------------------------------------


def _cluster_profile(hm, m200, zbin_idx, conc):
    """(rhos, rs, tau) arrays for cluster halos of mass m200 [Msun]."""
    m200 = np.asarray(m200, dtype=float)
    rhoc = hm.rhoc[zbin_idx]
    c = np.full_like(m200, conc)
    r200 = (3.0 * m200 / (4.0 * np.pi * 200.0 * rhoc)) ** (1.0 / 3.0)
    rs = r200 / c
    mu = np.log(1.0 + c) - c / (1.0 + c)
    rhos = (200.0 / 3.0) * rhoc * c ** 3 / mu
    return rhos, rs, c


class ClusterKappa:
    """Convergence from a fixed set of cluster halos, evaluable at any
    sight line, with optional mass/centering perturbations for the MC."""

    def __init__(self, cfg, hm, cosmo, clusters: pd.DataFrame):
        self.cfg = cfg
        self.hm = hm
        self.df = clusters.reset_index(drop=True)
        for col in ("ra", "dec", "z", "m200"):  # guard object dtypes
            self.df[col] = self.df[col].astype(float)
        self.ibin = hm.zbin_index(self.df["z"].to_numpy())
        self.sigcr = sigma_crit_msun_mpc2(
            cosmo, hm.zbins[self.ibin], cfg.source.z_src)
        self.conc = cfg.clusters.concentration

    def kappa(self, ra0, dec0, dlogm=None, dra_arcsec=None,
              ddec_arcsec=None):
        """Per-cluster kappa at (ra0, dec0). dlogm/offsets: per-cluster
        perturbations (mass scatter, miscentering) for MC draws."""
        if len(self.df) == 0:
            return np.zeros(0)
        ra = self.df["ra"].to_numpy().copy()
        dec = self.df["dec"].to_numpy().copy()
        if dra_arcsec is not None:
            ra = ra + dra_arcsec / 3600.0 / np.cos(np.radians(dec))
            dec = dec + ddec_arcsec / 3600.0
        m200 = self.df["m200"].to_numpy()
        if dlogm is not None:
            m200 = m200 * 10.0 ** np.asarray(dlogm)
        rhos, rs, tau = _cluster_profile(self.hm, m200, self.ibin, self.conc)
        theta = angular_sep_arcsec(ra0, dec0, ra, dec)
        theta_s = rs / self.hm.da[self.ibin] * 206264.806
        x = np.clip(theta / theta_s, 1e-3, None)
        sig = self.hm.sigma_dimless(x, tau)
        return rhos * rs * sig / self.sigcr

    def kappa_sum(self, ra0, dec0, **kw):
        return float(self.kappa(ra0, dec0, **kw).sum())

    def mc_kappa_sum(self, ra0, dec0, rng, n_mc):
        """MC draws of the summed kappa: lognormal mass scatter + Rayleigh
        miscentering per cluster per draw."""
        ccfg = self.cfg.clusters
        n_cl = len(self.df)
        out = np.zeros(n_mc)
        if n_cl == 0:
            return out
        for i in range(n_mc):
            dlogm = rng.normal(0.0, ccfg.mass_scatter_dex, n_cl)
            r_mis = rng.rayleigh(ccfg.miscentering_arcmin * 60.0, n_cl)
            phi = rng.uniform(0, 2 * np.pi, n_cl)
            out[i] = self.kappa(
                ra0, dec0, dlogm=dlogm,
                dra_arcsec=r_mis * np.cos(phi),
                ddec_arcsec=r_mis * np.sin(phi)).sum()
        return out


def assign_members(cfg, df_gal: pd.DataFrame, clusters: pd.DataFrame,
                   hm) -> np.ndarray:
    """Boolean mask over df_gal: galaxy belongs to a catalog cluster
    (within theta_200 and a redshift window); their individual halos are
    REPLACED by the cluster halo."""
    mask = np.zeros(len(df_gal), dtype=bool)
    if len(clusters) == 0:
        return mask
    ra_g = df_gal["ra"].to_numpy()
    dec_g = df_gal["dec"].to_numpy()
    z_spec = df_gal["z_spec"].to_numpy(dtype=float)
    zp = df_gal["zp_med"].to_numpy(dtype=float)
    sig_eff = 0.5 * ((df_gal["zp_med"] - df_gal["zp_l68"]).clip(lower=0.02)
                     + (df_gal["zp_u68"] - df_gal["zp_med"]).clip(lower=0.02)
                     ).to_numpy(dtype=float)

    ibin = hm.zbin_index(clusters["z"].to_numpy())
    rhoc = hm.rhoc[ibin]
    r200 = (3.0 * clusters["m200"].to_numpy()
            / (4.0 * np.pi * 200.0 * rhoc)) ** (1.0 / 3.0)
    theta200 = r200 / hm.da[ibin] * 206264.806  # arcsec

    dz_max = cfg.clusters.member_dz
    for k in range(len(clusters)):
        zc = float(clusters["z"].iloc[k])
        sep = angular_sep_arcsec(float(clusters["ra"].iloc[k]),
                                 float(clusters["dec"].iloc[k]), ra_g, dec_g)
        in_r = sep < theta200[k]
        spec_m = np.isfinite(z_spec) & (np.abs(z_spec - zc)
                                        < dz_max * (1 + zc))
        phot_m = ~np.isfinite(z_spec) & (np.abs(zp - zc)
                                         < np.maximum(dz_max * (1 + zc),
                                                      sig_eff))
        mask |= in_r & (spec_m | phot_m)
    return mask


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def m200_from_sigma(sigma_v, z, cosmo):
    """Evrard et al. 2008 sigma-M200 relation (dark-matter calibrated)."""
    hz = cosmo.H(z).value / 100.0
    return 1e15 / hz * (sigma_v / 1082.9) ** (1.0 / 0.3361)


def velocity_dispersion(cfg, df_gal, ra, dec, z, rmax_arcmin=6.0):
    """Biweight velocity dispersion of spec members around a center;
    returns dict with N, sigma_v [km/s], and the implied M200."""
    sep = angular_sep_arcsec(ra, dec, df_gal["ra"].to_numpy(),
                             df_gal["dec"].to_numpy())
    zs = df_gal["z_spec"].to_numpy(dtype=float)
    m = (sep < rmax_arcmin * 60.0) & np.isfinite(zs) \
        & (np.abs(zs - z) < 0.01 * (1 + z))
    zz = zs[m]
    for _ in range(3):
        if zz.size < 5:
            break
        v = C_KMS * (zz - np.median(zz)) / (1 + np.median(zz))
        sig = max(biweight_scale(v), 100.0)
        zz = zz[np.abs(v) < 3.0 * sig]
    if zz.size < 4:
        return {"n_members": int(zz.size), "sigma_v_kms": np.nan,
                "m200_dyn": np.nan}
    v = C_KMS * (zz - np.median(zz)) / (1 + np.median(zz))
    sig = float(biweight_scale(v))
    return {"n_members": int(zz.size), "sigma_v_kms": round(sig, 0),
            "m200_dyn": float(m200_from_sigma(sig, z, cfg.cosmo)),
            "rmax_arcmin": rmax_arcmin}
