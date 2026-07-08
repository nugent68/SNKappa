#!/usr/bin/env python
"""DES-SN5YR full-survey lensing analysis: predicted per-SN kappa vs
Hubble residuals across all ten DES SN fields.

Upgrades over scripts/des_pilot.py:
- all four field groups (X, S, C, E; the 194 low-z external SNe are skipped:
  sigma_lens < 5 mmag there),
- deeper foreground catalog (dereddened z <= 22.5; LS DR10-south depth),
- finer source-redshift bins (0.05),
- miscentering + richness-scatter MARGINALIZED cluster term (the pilot
  showed face-value cluster-core kappa dilutes the regression),
- restructured batch engine: per-galaxy halo tables are z_src-independent
  and built ONCE per region; only Sigma_crit and photo-z truncation are
  recomputed per bin.

Run: .venv/bin/python scripts/des_full.py   (from the SNKappa directory)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.config import (Config, ClustersConfig, CosmologyConfig,
                            DataConfig, DeflectorConfig, HaloModelConfig,
                            LensGroupConfig, LosConfig, MonteCarloConfig,
                            OutputConfig, RandomsConfig, SourceConfig)
from snkappa import catalog, clusters as clu, photoz
from snkappa.datalab import TapClient
from snkappa.halos import HaloModel
from snkappa.kappa import angular_sep_arcsec, sigma_crit_msun_mpc2
from snkappa.stellar import make_estimator

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

FIELD_GROUPS = {
    "X": (("X1", "X2", "X3"), (35.6, -5.3)),
    "S": (("S1", "S2"), (42.0, -0.5)),
    "C": (("C1", "C2", "C3"), (53.7, -28.1)),
    "E": (("E1", "E2"), (8.7, -43.5)),
}
MAG_LIMIT = 22.5
APERTURE_ARCMIN = 10.0
R_IN = 3.0                      # arcsec
ZSRC_EDGES = np.arange(0.10, 1.1501, 0.05)
ZC = np.arange(0.02, 1.14, 0.04)   # photo-z marginalization grid
N_RAND = 120
CL_ND = 64                      # cluster MC draws (miscentering + mass)
CL_MIS_ARCSEC = 30.0
CL_MSCAT = 0.25
OMEGA_M = 0.352


def read_snana(path):
    rows, names = [], None
    for line in open(path):
        if line.startswith("VARNAMES:"):
            names = line.split()[1:]
        elif line.startswith("SN:"):
            rows.append(line.split()[1:])
    return pd.DataFrame(rows, columns=names)


def load_des():
    hd = read_snana("des_pilot/DES_HD.csv")
    meta = read_snana("des_pilot/DES_meta.csv")
    sn = hd.merge(meta[["CID", "FIELD", "HOST_RA", "HOST_DEC"]], on="CID")
    for c in ("zHD", "MU", "MUERR", "PROBIA_BEAMS", "HOST_RA", "HOST_DEC"):
        sn[c] = pd.to_numeric(sn[c], errors="coerce")
    sn = sn[np.isfinite(sn.zHD) & np.isfinite(sn.MU) & np.isfinite(sn.HOST_RA)
            & (sn.HOST_RA > -100) & (sn.zHD > 0.1) & (sn.zHD < 1.15)]
    return sn.reset_index(drop=True)


def make_cfg(center, region_annulus_outer):
    return Config(
        source=SourceConfig(name="DES", ra_src=center[0], dec_src=center[1],
                            z_src=1.15),
        deflector=DeflectorConfig(ra_lens=center[0], dec_lens=center[1],
                                  z_lens=0.3, r_exclude_arcsec=R_IN),
        lens_group=LensGroupConfig(r_group_arcmin=0.01),
        los=LosConfig(aperture_radius_arcmin=APERTURE_ARCMIN,
                      mag_limit=MAG_LIMIT),
        randoms=RandomsConfig(n_random_los=N_RAND,
                              annulus_deg=[0.5, region_annulus_outer]),
        montecarlo=MonteCarloConfig(n_mc=64, seed=20130901),
        data=DataConfig(), halo_model=HaloModelConfig(),
        cosmology=CosmologyConfig(), clusters=ClustersConfig(enabled=True),
        output=OutputConfig(dir="output/des_full"),
    )


class BatchEngine:
    """Halo tables built once; Sigma_crit / photo-z truncation set per z_src."""

    def __init__(self, cfg, df, hm, est):
        self.hm = hm
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
        self.s_ths = rs / hm.da[ib] * 206264.806
        self.s_tau = tau

        self.p_ra = p.ra.to_numpy(); self.p_dec = p.dec.to_numpy()
        n = len(p)
        izc = hm.zbin_index(ZC)
        self.p_amp0 = np.empty((n, ZC.size), np.float32)
        self.p_ths = np.empty((n, ZC.size), np.float32)
        self.p_tau = np.empty((n, ZC.size), np.float32)
        magp = {b: p[f"mag_{b}"].to_numpy(float)
                for b in ("g", "r", "i", "z", "w1") if f"mag_{b}" in p}
        for j, zc in enumerate(ZC):
            lm = est.logmstar(magp, np.full(n, zc))
            rhos, rs, tau = hm.halo_params(lm, np.full(n, izc[j]))
            self.p_amp0[:, j] = rhos * rs
            self.p_ths[:, j] = rs / hm.da[izc[j]] * 206264.806
            self.p_tau[:, j] = tau
        mu, slo, shi = photoz.split_normal_params(
            p.zp_med.to_numpy(float), p.zp_l68.to_numpy(float),
            p.zp_u68.to_numpy(float), p.zp_std.to_numpy(float))
        dzc = np.gradient(ZC)
        self.p_pdfw = (photoz.grid_pdf(ZC, mu, slo, shi)
                       * dzc[None, :]).astype(np.float32)
        self.izc = izc

    def set_zsrc(self, cosmo, z_src):
        sig_fine = sigma_crit_msun_mpc2(cosmo, self.hm.zbins, z_src)
        self.sA = np.where(self.s_z < z_src - 0.01,
                           self.s_amp0 / sig_fine[self.s_ib], 0.0)
        sig_zc = sigma_crit_msun_mpc2(cosmo, ZC, z_src)
        w = self.p_pdfw * (ZC < z_src - 0.01)[None, :]
        with np.errstate(divide="ignore"):
            self.pWA = (w * (self.p_amp0 / sig_zc[None, :])).astype(np.float32)

    def kappa_gal(self, ra0, dec0, r_out):
        th = angular_sep_arcsec(ra0, dec0, self.s_ra, self.s_dec)
        m = (th > R_IN) & (th < r_out) & (self.sA > 0)
        k = float(np.sum(self.sA[m] * self.hm.sigma_dimless(
            th[m] / self.s_ths[m], self.s_tau[m]))) if m.any() else 0.0
        th = angular_sep_arcsec(ra0, dec0, self.p_ra, self.p_dec)
        m = (th > R_IN) & (th < r_out)
        if m.any():
            x = th[m][:, None] / self.p_ths[m]
            sig = self.hm.sigma_dimless(
                x.ravel(), self.p_tau[m].ravel()).reshape(x.shape)
            k += float(np.sum(self.pWA[m] * sig))
        return k


def cluster_kappa_marg(ra0, dec0, z_src, cl, hm, cosmo, rng):
    """Miscentering + mass-scatter marginalized cluster term (fast, local)."""
    dx = (cl.ra.to_numpy() - ra0) * np.cos(np.radians(dec0)) * 3600.0
    dy = (cl.dec.to_numpy() - dec0) * 3600.0
    sep = np.hypot(dx, dy)
    near = (sep < 1800.0) & (cl.z.to_numpy() < z_src - 0.02)
    if not near.any():
        return 0.0
    dx, dy = dx[near], dy[near]
    zc = cl.z.to_numpy()[near]
    m0 = cl.m200.to_numpy()[near]
    n = zc.size
    ib = hm.zbin_index(zc)
    rhoc = hm.rhoc[ib]
    sigcr = sigma_crit_msun_mpc2(cosmo, hm.zbins[ib], z_src)
    r = rng.rayleigh(CL_MIS_ARCSEC, (CL_ND, n))
    ph = rng.uniform(0, 2 * np.pi, (CL_ND, n))
    sepd = np.hypot(dx[None, :] + r * np.cos(ph), dy[None, :] + r * np.sin(ph))
    m200 = m0[None, :] * 10.0 ** rng.normal(0, CL_MSCAT, (CL_ND, n))
    cc = 5.0
    r200 = (3 * m200 / (4 * np.pi * 200 * rhoc[None, :])) ** (1 / 3)
    rs = r200 / cc
    rhos = (200 / 3) * rhoc * cc**3 / (np.log(1 + cc) - cc / (1 + cc))
    ths = rs / hm.da[ib][None, :] * 206264.806
    x = np.clip(sepd / ths, 1e-3, None)
    sig = hm.sigma_dimless(x.ravel(), np.full(x.size, cc)).reshape(x.shape)
    kap = rhos[None, :] * rs * sig / sigcr[None, :]
    return float(kap.sum(axis=1).mean())


def main():
    outdir = Path("output/des_full"); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20130901)
    sn_all = load_des()
    r_out = APERTURE_ARCMIN * 60.0
    results = []

    for gname, (fields, center) in FIELD_GROUPS.items():
        sn = sn_all[sn_all.FIELD.isin(fields)].reset_index(drop=True)
        if len(sn) == 0:
            continue
        rad = angular_sep_arcsec(center[0], center[1],
                                 sn.HOST_RA.to_numpy(),
                                 sn.HOST_DEC.to_numpy()).max() / 3600.0
        cfg = make_cfg(center, rad + 0.25)
        log(f"=== {gname} fields {fields}: {len(sn)} SNe, "
            f"region radius {rad + 0.25 + APERTURE_ARCMIN/60:.2f} deg ===")
        tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
        df = catalog.clean_and_merge(cfg, *catalog.fetch_regional(cfg, tap))
        n_spec = int(df.z_spec.notna().sum())
        log(f"  catalog: {len(df)} galaxies ({n_spec} spec-z)")
        cl = clu.fetch_clusters(cfg, lambda u: TapClient(u, cfg.data.cache_dir))
        hm = HaloModel(cfg.halo_model, cfg.cosmo, 1.15)
        members = clu.assign_members(cfg, df, cl, hm)
        df = df[~members].reset_index(drop=True)
        log(f"  {len(cl)} clusters; {int(members.sum())} members replaced")
        est = make_estimator("nir1um", cfg.cosmo)
        eng = BatchEngine(cfg, df, hm, est)
        log(f"  halo tables built "
            f"({(eng.p_amp0.nbytes*3 + eng.p_pdfw.nbytes)/1e6:.0f} MB)")

        th = np.sqrt(rng.uniform(0, 1, N_RAND)) * (rad - 0.1)
        ph = rng.uniform(0, 2 * np.pi, N_RAND)
        rra = center[0] + th * np.cos(ph) / np.cos(np.radians(center[1]))
        rdec = center[1] + th * np.sin(ph)

        for lo, hi in zip(ZSRC_EDGES[:-1], ZSRC_EDGES[1:]):
            in_bin = sn[(sn.zHD >= lo) & (sn.zHD < hi)]
            if len(in_bin) == 0:
                continue
            z_src = 0.5 * (lo + hi)
            eng.set_zsrc(cfg.cosmo, z_src)

            def kap(ra, dec):
                return (eng.kappa_gal(ra, dec, r_out)
                        + cluster_kappa_marg(ra, dec, z_src, cl, hm,
                                             cfg.cosmo, rng))

            k_rand = np.array([kap(a, d) for a, d in zip(rra, rdec)])
            zp = k_rand.mean()
            sig = 0.5 * np.subtract(*np.percentile(k_rand, [84, 16]))
            for _, row in in_bin.iterrows():
                k = kap(row.HOST_RA, row.HOST_DEC)
                results.append({
                    "CID": row.CID, "FIELD": row.FIELD, "GROUP": gname,
                    "zHD": row.zHD, "MU": row.MU, "MUERR": row.MUERR,
                    "PROBIA": row.PROBIA_BEAMS, "kappa_raw": k,
                    "kappa_ext": k - zp, "zbin": z_src,
                    "rand_mean": zp, "rand_sig": sig, "n_spec_region": n_spec})
            log(f"  z[{lo:.2f},{hi:.2f}): {len(in_bin):3d} SNe | "
                f"zp {zp:.4f} sig {sig:.4f}")

    res = pd.DataFrame(results)
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=OMEGA_M)
    res["hr"] = res.MU - cosmo.distmod(res.zHD.to_numpy()).value
    w = 1.0 / res.MUERR**2
    res["hr"] -= np.average(res.hr, weights=w)
    res.to_csv(outdir / "des_all_kappa.csv", index=False)

    def fit(d, label):
        x = d.kappa_ext.to_numpy(); y = d.hr.to_numpy()
        ww = 1 / d.MUERR.to_numpy()**2
        b = np.polyfit(x, y, 1, w=ww)[0]
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(d), len(d))
            boot.append(np.polyfit(x[i], y[i], 1, w=ww[i])[0])
        e = np.std(boot)
        log(f"{label:34s} N={len(d):4d} slope={b:+.2f}+-{e:.2f} "
            f"({abs(b)/e:.1f} sig) sig_k={x.std():.4f}")
        return b, e

    log("=" * 64)
    fit(res, "all fields, raw")
    good = res[res.PROBIA > 0.9]
    b, e = fit(good, "P(Ia)>0.9")
    fit(good[good.GROUP.isin(["X", "S"])], "P(Ia)>0.9, spec-rich (X+S)")
    fit(good[good.GROUP.isin(["C", "E"])], "P(Ia)>0.9, photo-only (C+E)")
    hi = good[good.kappa_ext > 0.005]; lo_ = good[good.kappa_ext <= 0.005]
    log(f"mean HR kappa>0.005: {np.average(hi.hr, weights=1/hi.MUERR**2):+.3f}"
        f" (N={len(hi)}) | kappa<=0.005: "
        f"{np.average(lo_.hr, weights=1/lo_.MUERR**2):+.3f} (N={len(lo_)})")
    log("top sightlines:")
    for _, r_ in good.nlargest(10, "kappa_ext").iterrows():
        log(f"  {r_.CID} ({r_.FIELD}) z={r_.zHD:.3f} "
            f"kappa={r_.kappa_ext:+.4f} HR={r_.hr:+.3f}")
    log(f"saved {outdir/'des_all_kappa.csv'}")


if __name__ == "__main__":
    main()
