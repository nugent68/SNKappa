#!/usr/bin/env python
"""DES-SN5YR full-survey lensing analysis: predicted per-SN kappa vs
Hubble residuals across all ten DES SN fields.

Pipeline (engine lives in snkappa.batch, TODO 3.9):
- all four field groups (X, S, C, E; the 194 low-z external SNe are skipped:
  sigma_lens < 5 mmag there),
- deeper foreground catalog (dereddened z <= 22.5; LS DR10-south depth),
- per-galaxy halo tables z_src-independent, built ONCE per region,
- deterministic miscentering- and mass-marginalized cluster tier
  (centered lognormal mass scatter, E[M] = M_catalog; physical
  miscentering scale 0.2 r500),
- kappa evaluated at the two source-redshift bin centers bracketing each
  SN's z_HD and linearly interpolated (removes the +/-0.025 z_src
  quantization),
- random sightlines with a low-count (masking / survey edge) guard;
  per-SN unmasked-area fraction flag,
- prediction and Hubble residuals use the SAME cosmology
  (flat LCDM, H0=70, Om0=0.352, the DES-SN5YR SN-only fit),
- inverse-variance regression via np.polyfit with w = 1/sigma
  (the polyfit weight multiplies the unsquared residual).

Run: .venv/bin/python scripts/des_full.py            (headline)
Variants (robustness / systematics; see TODO_REVIEW.md):
  --variant excise   --excise-host      host-environment excision (1.2a;
                                        galaxies AND clusters near the plane)
  --variant nospecz  --no-specz         ignore DESI spec-z (1.3)
  --variant w1only   --require-w1       drop Taylor2011 fallback pop. (1.5)
  --variant naive    --smhm-inverse naive   legacy SMHM inversion (2.2)
  --variant cap14.1  --logmh-max 14.1   cap sensitivity (2.3)
"""

import argparse
import json
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
from snkappa import catalog, clusters as clu
from snkappa.batch import BatchEngine, ClusterField
from snkappa.datalab import TapClient
from snkappa.fitting import bootstrap_slope
from snkappa.halos import HaloModel
from snkappa.kappa import angular_sep_arcsec
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
ZSRC_CENTERS = 0.5 * (ZSRC_EDGES[:-1] + ZSRC_EDGES[1:])
ZC = np.arange(0.02, 1.14, 0.04)   # photo-z marginalization grid
OMEGA_M = 0.352                 # DES-SN5YR flat-LCDM (SN-only); used for BOTH
H0 = 70.0                       # the kappa prediction and the residuals
AREA_FLAG_MIN = 0.90            # per-SN unmasked-area fraction threshold


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="", help="output suffix")
    p.add_argument("--excise-host", action="store_true",
                   help="excise foregrounds with |z-z_src| < 0.1(1+z_src)")
    p.add_argument("--no-specz", action="store_true",
                   help="ignore DESI spec-z entirely (photo-z only)")
    p.add_argument("--require-w1", action="store_true",
                   help="drop galaxies without a W1 detection")
    p.add_argument("--smhm-inverse", default="posterior",
                   choices=("posterior", "naive"))
    p.add_argument("--logmh-max", type=float, default=13.8)
    p.add_argument("--mstar-offset", type=float, default=0.0,
                   help="global logM* offset in dex (M*/L calibration "
                        "systematic for the error budget)")
    p.add_argument("--mstar-method", default="nir1um_fsf",
                   help="stellar-mass estimator (nir1um_fsf = rest-1um "
                        "recalibrated to the DESI DR1 FastSpecFit scale; "
                        "nir1um = legacy constant M*/L)")
    p.add_argument("--n-rand", type=int, default=500)
    p.add_argument("--groups", default="X,S,C,E")
    return p.parse_args()


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
    sn = hd.merge(meta[["CID", "FIELD", "HOST_RA", "HOST_DEC",
                        "HOST_LOGMASS"]], on="CID")
    for c in ("zHD", "MU", "MUERR", "PROBIA_BEAMS", "HOST_RA", "HOST_DEC",
              "HOST_LOGMASS"):
        sn[c] = pd.to_numeric(sn[c], errors="coerce")
    sn = sn[np.isfinite(sn.zHD) & np.isfinite(sn.MU) & np.isfinite(sn.HOST_RA)
            & (sn.HOST_RA > -100) & (sn.zHD > 0.1) & (sn.zHD < 1.15)]
    return sn.reset_index(drop=True)


def make_cfg(args, center, region_annulus_outer):
    return Config(
        source=SourceConfig(name="DES", ra_src=center[0], dec_src=center[1],
                            z_src=1.15),
        deflector=DeflectorConfig(ra_lens=center[0], dec_lens=center[1],
                                  z_lens=0.3, r_exclude_arcsec=R_IN),
        lens_group=LensGroupConfig(r_group_arcmin=0.01),
        los=LosConfig(aperture_radius_arcmin=APERTURE_ARCMIN,
                      mag_limit=MAG_LIMIT),
        randoms=RandomsConfig(n_random_los=args.n_rand,
                              annulus_deg=[0.5, region_annulus_outer]),
        montecarlo=MonteCarloConfig(n_mc=64, seed=20130901),
        data=DataConfig(),
        halo_model=HaloModelConfig(smhm_inverse=args.smhm_inverse,
                                   logmh_max=args.logmh_max),
        cosmology=CosmologyConfig(H0=H0, Om0=OMEGA_M),
        clusters=ClustersConfig(enabled=True),
        output=OutputConfig(dir="output/des_full"),
    )


def main():
    args = parse_args()
    suffix = f"_{args.variant}" if args.variant else ""
    excise_frac = 0.1 if args.excise_host else None
    outdir = Path("output/des_full"); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20130901)
    sn_all = load_des()
    r_out = APERTURE_ARCMIN * 60.0
    results = []

    for gname in args.groups.split(","):
        fields, center = FIELD_GROUPS[gname]
        sn = sn_all[sn_all.FIELD.isin(fields)].reset_index(drop=True)
        if len(sn) == 0:
            continue
        rad = angular_sep_arcsec(center[0], center[1],
                                 sn.HOST_RA.to_numpy(),
                                 sn.HOST_DEC.to_numpy()).max() / 3600.0
        cfg = make_cfg(args, center, rad + 0.25)
        log(f"=== {gname} fields {fields}: {len(sn)} SNe, "
            f"region radius {rad + 0.25 + APERTURE_ARCMIN/60:.2f} deg ===")
        tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
        df_raw, zpix = catalog.fetch_regional(cfg, tap)
        if args.no_specz:
            zpix = zpix.iloc[0:0]     # variant 1.3: pretend DESI never existed
        df = catalog.clean_and_merge(cfg, df_raw, zpix)
        n_spec = int(df.z_spec.notna().sum())
        log(f"  catalog: {len(df)} galaxies ({n_spec} spec-z)")
        cl = clu.fetch_clusters(cfg, lambda u: TapClient(u, cfg.data.cache_dir))
        hm = HaloModel(cfg.halo_model, cfg.cosmo, 1.15)
        members = clu.assign_members(cfg, df, cl, hm)
        df = df[~members].reset_index(drop=True)
        log(f"  {len(cl)} clusters; {int(members.sum())} members replaced")
        est = make_estimator(args.mstar_method, cfg.cosmo)
        if args.mstar_offset:
            class _Offset:
                def __init__(self, base, off):
                    self.base, self.off = base, off
                def logmstar(self, mags, z):
                    return self.base.logmstar(mags, z) + self.off
            est = _Offset(est, args.mstar_offset)
        eng = BatchEngine(cfg, df, hm, est, ZC, R_IN,
                          require_w1=args.require_w1)
        clf = ClusterField(cl, hm,
                           mass_scatter_dex=cfg.clusters.mass_scatter_dex,
                           miscenter_frac_r500=cfg.clusters.miscenter_frac_r500,
                           conc=cfg.clusters.concentration)
        log(f"  halo tables built ({eng.table_bytes()/1e6:.0f} MB); "
            f"cluster profiles precomputed")

        # random sightlines, drawn once per group; low-count guard (TODO 0.4)
        n_rand = args.n_rand
        th = np.sqrt(rng.uniform(0, 1, n_rand)) * (rad - 0.1)
        ph = rng.uniform(0, 2 * np.pi, n_rand)
        rra = center[0] + th * np.cos(ph) / np.cos(np.radians(center[1]))
        rdec = center[1] + th * np.sin(ph)
        ngal_rand = np.array([eng.counts(a, d, r_out)
                              for a, d in zip(rra, rdec)])
        ok = ngal_rand > 0.5 * np.median(ngal_rand)
        log(f"  randoms: {int(ok.sum())}/{n_rand} pass the count guard "
            f"(median ngal {int(np.median(ngal_rand))})")
        rra, rdec = rra[ok], rdec[ok]

        # per-SN interpolation assignments (TODO 3.4): each SN is evaluated
        # at the two bracketing bin centers and linearly interpolated
        zc0, dz = ZSRC_CENTERS[0], ZSRC_CENTERS[1] - ZSRC_CENTERS[0]
        pos = np.clip((sn.zHD.to_numpy() - zc0) / dz, 0.0,
                      ZSRC_CENTERS.size - 1.0)
        k_lo = np.floor(pos).astype(int)
        k_hi = np.minimum(k_lo + 1, ZSRC_CENTERS.size - 1)
        t_int = pos - k_lo
        t_int[k_hi == k_lo] = 0.0

        kap_sn = np.full((len(sn), 2), np.nan)   # [SN, (lo, hi)]
        zp_c = np.full(ZSRC_CENTERS.size, np.nan)
        sig_c = np.full(ZSRC_CENTERS.size, np.nan)

        for k, z_src in enumerate(ZSRC_CENTERS):
            need_lo = np.flatnonzero(k_lo == k)
            need_hi = np.flatnonzero((k_hi == k) & (k_lo != k))
            if need_lo.size + need_hi.size == 0:
                continue
            eng.set_zsrc(cfg.cosmo, z_src, excise_frac=excise_frac)
            clf.set_zsrc(cfg.cosmo, z_src, excise_frac=excise_frac)

            def kap(ra, dec):
                return eng.kappa_gal(ra, dec, r_out) + clf.kappa_sum(ra, dec)

            k_rand = np.array([kap(a, d) for a, d in zip(rra, rdec)])
            zp_c[k] = k_rand.mean()
            sig_c[k] = 0.5 * np.subtract(*np.percentile(k_rand, [84, 16]))
            for i in np.concatenate([need_lo, need_hi]):
                col = 0 if k_lo[i] == k else 1
                kap_sn[i, col] = kap(sn.HOST_RA.iloc[i], sn.HOST_DEC.iloc[i])
            log(f"  z_src={z_src:.3f}: {need_lo.size + need_hi.size:3d} SN "
                f"evals | zp {zp_c[k]:.4f} sig {sig_c[k]:.4f}")

        for i, row in sn.iterrows():
            t = t_int[i]
            lo, hi = k_lo[i], k_hi[i]
            k_raw = (1 - t) * kap_sn[i, 0] + t * (kap_sn[i, 1]
                                                  if t > 0 else 0.0)
            zp = (1 - t) * zp_c[lo] + t * (zp_c[hi] if t > 0 else 0.0)
            sig = (1 - t) * sig_c[lo] + t * (sig_c[hi] if t > 0 else 0.0)
            afrac, _ = catalog.area_fraction(cfg, df, row.HOST_RA,
                                             row.HOST_DEC)
            # cluster-plane confounder guard: min |z_cl - z_SN|/(1+z_SN)
            # over catalog clusters within 2 arcmin of the sightline
            # (NaN if none) -- feeds the no-cluster-at-SN-plane robustness row
            cl_dz = np.nan
            if len(cl):
                sepc = angular_sep_arcsec(row.HOST_RA, row.HOST_DEC,
                                          cl.ra.to_numpy(), cl.dec.to_numpy())
                nearc = sepc < 120.0
                if nearc.any():
                    cl_dz = float(np.min(np.abs(cl.z.to_numpy()[nearc]
                                                - row.zHD)) / (1.0 + row.zHD))
            results.append({
                "CID": row.CID, "FIELD": row.FIELD, "GROUP": gname,
                "zHD": row.zHD, "MU": row.MU, "MUERR": row.MUERR,
                "PROBIA": row.PROBIA_BEAMS,
                "HOST_RA": row.HOST_RA, "HOST_DEC": row.HOST_DEC,
                "HOST_LOGMASS": row.HOST_LOGMASS,
                "kappa_raw": k_raw, "kappa_ext": k_raw - zp,
                "zbin": ZSRC_CENTERS[lo if t < 0.5 else hi],
                "rand_mean": zp, "rand_sig": sig,
                "n_rand_ok": int(rra.size),
                "area_frac": afrac, "area_flag": afrac < AREA_FLAG_MIN,
                "cl_dz_min_2am": cl_dz,
                "n_spec_region": n_spec})

    res = pd.DataFrame(results)
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=OMEGA_M)
    res["hr"] = res.MU - cosmo.distmod(res.zHD.to_numpy()).value
    w = 1.0 / res.MUERR**2
    res["hr_glob"] = res.hr - np.average(res.hr, weights=w)
    # per-z-bin demean (TODO 3.1): kills residual cosmology/calibration
    # z-trends; kappa_ext is already per-bin zero-pointed, so this is free
    res["hr"] = res.hr_glob
    for zb, idx in res.groupby("zbin").groups.items():
        ww = 1.0 / res.loc[idx, "MUERR"] ** 2
        res.loc[idx, "hr"] -= np.average(res.loc[idx, "hr"], weights=ww)
    res.to_csv(outdir / f"des_all_kappa{suffix}.csv", index=False)

    def fit(d, label):
        b, e = bootstrap_slope(d.kappa_ext.to_numpy(), d.hr.to_numpy(),
                               d.MUERR.to_numpy(), rng)
        log(f"{label:34s} N={len(d):4d} slope={b:+.2f}+-{e:.2f} "
            f"({abs(b)/e:.1f} sig) sig_k={d.kappa_ext.std():.4f}")
        return b, e

    log("=" * 64)
    summary = {"variant": args.variant or "headline",
               "args": vars(args), "n_all": len(res)}
    summary["slope_all"] = fit(res, "all fields, raw")
    good = res[res.PROBIA > 0.9]
    summary["n_good"] = len(good)
    summary["slope_good"] = fit(good, "P(Ia)>0.9")
    if {"X", "S"} & set(good.GROUP.unique()):
        summary["slope_XS"] = fit(good[good.GROUP.isin(["X", "S"])],
                                  "P(Ia)>0.9, spec-rich (X+S)")
    if {"C", "E"} & set(good.GROUP.unique()):
        summary["slope_CE"] = fit(good[good.GROUP.isin(["C", "E"])],
                                  "P(Ia)>0.9, photo-only (C+E)")
    hi = good[good.kappa_ext > 0.005]; lo_ = good[good.kappa_ext <= 0.005]
    log(f"mean HR kappa>0.005: {np.average(hi.hr, weights=1/hi.MUERR**2):+.3f}"
        f" (N={len(hi)}) | kappa<=0.005: "
        f"{np.average(lo_.hr, weights=1/lo_.MUERR**2):+.3f} (N={len(lo_)})")
    log(f"area flag: {int(res.area_flag.sum())}/{len(res)} SNe below "
        f"{AREA_FLAG_MIN:.0%} unmasked-area fraction")
    log("top sightlines:")
    for _, r_ in good.nlargest(10, "kappa_ext").iterrows():
        log(f"  {r_.CID} ({r_.FIELD}) z={r_.zHD:.3f} "
            f"kappa={r_.kappa_ext:+.4f} HR={r_.hr:+.3f}")
    (outdir / f"fit_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, default=float))
    log(f"saved {outdir / f'des_all_kappa{suffix}.csv'}")


if __name__ == "__main__":
    main()
