#!/usr/bin/env python
"""Union3 predicted-kappa survey run (build Phase 3).

Mirrors scripts/des_full.py (post-9e3a77e: gal/cl decomposition, shear,
exact dmu_pred, cluster-proximity and area flags) but iterates the
FoF region table from snkappa.regions instead of hard-coded field
groups, with z_src extended to 1.60 for the Union3 redshift reach.

Per region: LS regional catalog -> Wen & Han clusters (VizieR; on
failure the region runs without the cluster tier, flagged) -> member
replacement -> BatchEngine + ClusterField -> randoms with count guard
-> per-z_src-bin evaluation, interpolated to each SN's z.

Fits exclude clipped SNe and the cluster-targeted See-Change sample
(selection on kappa); both still receive predictions (flags carried).

Run: .venv/bin/python scripts/union3_full.py     (after union3_prep.py)
Outputs: output/union3/union3_kappa.csv, regions.csv, fit_summary.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", message=".*Second redshift.*")

from snkappa.config import (Config, ClustersConfig, CosmologyConfig,
                            DataConfig, DeflectorConfig, HaloModelConfig,
                            LensGroupConfig, LosConfig, MonteCarloConfig,
                            OutputConfig, RandomsConfig, SourceConfig)
from snkappa import catalog, clusters as clu
from snkappa.batch import BatchEngine, ClusterField
from snkappa.datalab import TapClient
from snkappa.fitting import bootstrap_slope, two_component_slopes
from snkappa.halos import HaloModel
from snkappa.kappa import angular_sep_arcsec
from snkappa.regions import build_regions
from snkappa.stellar import make_estimator

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

HD_CSV = Path("output/union3/union3_hd.csv")
OUTDIR = Path("output/union3")
Z_SRC_MAX = 1.60
Z_MIN = 0.1
MAG_LIMIT = 22.5
APERTURE_ARCMIN = 10.0
R_IN = 3.0
ZSRC_EDGES = np.arange(0.10, Z_SRC_MAX + 1e-4, 0.05)
ZSRC_CENTERS = 0.5 * (ZSRC_EDGES[:-1] + ZSRC_EDGES[1:])
ZC = np.arange(0.02, Z_SRC_MAX - 0.01, 0.04)
OM_FID = 0.356
AREA_FLAG_MIN = 0.90
N_CLIP = [0]
SLOPE_TH = -5.0 / np.log(10.0)


def dmu_exact(kappa, gamma):
    arg = (1.0 - np.asarray(kappa, float)) ** 2 \
        - np.asarray(gamma, float) ** 2
    n_bad = int(np.count_nonzero(arg <= 1e-4))
    if n_bad:
        N_CLIP[0] += n_bad
    return 2.5 * np.log10(np.clip(arg, 1e-4, None))


WH_LOCAL = Path("data_union3/wenhan24_table2.dat.gz")
_WH_CACHE = [None]


def local_clusters(center, radius_deg):
    """Wen & Han 2024 clusters from the full CDS table (local cone;
    VizieR-independent -- the TAP service 503'd through the smoke test).
    Same columns/conversions as clusters.fetch_clusters (M200 = 1.4 M500)."""
    if _WH_CACHE[0] is None:
        wh = pd.read_fwf(WH_LOCAL, compression="gzip",
                         colspecs=[(11, 27), (28, 37), (38, 47),
                                   (49, 55), (95, 100)],
                         names=["name", "ra", "dec", "z", "m500"])
        _WH_CACHE[0] = wh
        log(f"  loaded {len(wh)} Wen&Han clusters from local CDS table")
    wh = _WH_CACHE[0]
    pre = np.abs(wh.dec.to_numpy() - center[1]) < radius_deg + 0.05
    sub = wh[pre]
    sep = angular_sep_arcsec(center[0], center[1], sub.ra.to_numpy(),
                             sub.dec.to_numpy()) / 3600.0
    sub = sub[sep < radius_deg].copy()
    out = pd.DataFrame({"name": sub.name, "ra": sub.ra, "dec": sub.dec,
                        "z": sub.z, "m200": sub.m500 * 1e14 * 1.4})
    out = out[np.isfinite(out.m200) & (out.m200 > 0)
              & (out.z > 0) & (out.z < Z_SRC_MAX)]
    return out.reset_index(drop=True)


def make_cfg(center, annulus_outer, n_rand):
    return Config(
        source=SourceConfig(name="U3", ra_src=center[0], dec_src=center[1],
                            z_src=Z_SRC_MAX),
        deflector=DeflectorConfig(ra_lens=center[0], dec_lens=center[1],
                                  z_lens=0.3, r_exclude_arcsec=R_IN),
        lens_group=LensGroupConfig(r_group_arcmin=0.01),
        los=LosConfig(aperture_radius_arcmin=APERTURE_ARCMIN,
                      mag_limit=MAG_LIMIT),
        randoms=RandomsConfig(n_random_los=n_rand,
                              annulus_deg=[0.3, annulus_outer]),
        montecarlo=MonteCarloConfig(n_mc=64, seed=20130901),
        data=DataConfig(), halo_model=HaloModelConfig(),
        cosmology=CosmologyConfig(H0=70.0, Om0=OM_FID),
        clusters=ClustersConfig(enabled=True),
        output=OutputConfig(dir=str(OUTDIR)),
    )


MSTAR_METHOD = ["nir1um_fsf"]   # set from --mstar-method in main


def run_region(row, sn, hm, est, rng, results, failures):
    center = (row.ra, row.dec)
    rad = row.radius_deg
    n_rand = 500 if rad > 0.5 else 200
    cfg = make_cfg(center, max(rad + 0.25, 0.45), n_rand)
    tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
    df = catalog.clean_and_merge(cfg, *catalog.fetch_regional(cfg, tap))
    if MSTAR_METHOD[0].startswith("nir_direct"):
        from snkappa.nir import attach_nir
        df = attach_nir(df)
        log(f"  region {row.region}: deep-NIR attached "
            f"({df.attrs['n_nir_matched']}/{len(df)})")
    if WH_LOCAL.exists():
        cl = local_clusters(center, catalog.region_radius_deg(cfg))
        cl_ok = True
    else:
        try:
            cl = clu.fetch_clusters(cfg, lambda u: TapClient(
                u, cfg.data.cache_dir))
            cl_ok = True
        except Exception as exc:
            log(f"  region {row.region}: cluster fetch FAILED ({exc}); "
                f"running without cluster tier")
            cl = pd.DataFrame(columns=["name", "ra", "dec", "z", "m200"])
            cl_ok = False
    members = clu.assign_members(cfg, df, cl, hm) if len(cl) else \
        np.zeros(len(df), dtype=bool)
    df = df[~members].reset_index(drop=True)
    eng = BatchEngine(cfg, df, hm, est, ZC, R_IN)
    clf = ClusterField(cl, hm) if len(cl) else None

    rr = np.sqrt(rng.uniform(0, 1, n_rand)) * max(rad - 0.1, 0.35)
    ph = rng.uniform(0, 2 * np.pi, n_rand)
    rra = center[0] + rr * np.cos(ph) / np.cos(np.radians(center[1]))
    rdec = center[1] + rr * np.sin(ph)
    r_out = APERTURE_ARCMIN * 60.0
    ngal_rand = np.array([eng.counts(a, d, r_out)
                          for a, d in zip(rra, rdec)])
    ok = ngal_rand > 0.5 * np.median(ngal_rand)
    rra, rdec = rra[ok], rdec[ok]
    if rra.size == 0:
        raise RuntimeError("no valid randoms (masked/edge region: "
                           f"median aperture count "
                           f"{np.median(ngal_rand):.0f})")

    zc0, dz = ZSRC_CENTERS[0], ZSRC_CENTERS[1] - ZSRC_CENTERS[0]
    pos = np.clip((sn.zHD.to_numpy() - zc0) / dz, 0.0,
                  ZSRC_CENTERS.size - 1.0)
    k_lo = np.floor(pos).astype(int)
    k_hi = np.minimum(k_lo + 1, ZSRC_CENTERS.size - 1)
    t_int = pos - k_lo
    t_int[k_hi == k_lo] = 0.0

    n_sn = len(sn)
    kap_g = np.full((n_sn, 2), np.nan)
    kap_c = np.full((n_sn, 2), np.nan)
    gam = np.full((n_sn, 2), np.nan)
    dmu = np.full((n_sn, 2), np.nan)
    zp_c = np.full(ZSRC_CENTERS.size, np.nan)
    zp_g_c = np.full(ZSRC_CENTERS.size, np.nan)
    zp_cl_c = np.full(ZSRC_CENTERS.size, np.nan)
    zp_dmu_c = np.full(ZSRC_CENTERS.size, np.nan)
    sig_c = np.full(ZSRC_CENTERS.size, np.nan)

    for k, z_src in enumerate(ZSRC_CENTERS):
        need_lo = np.flatnonzero(k_lo == k)
        need_hi = np.flatnonzero((k_hi == k) & (k_lo != k))
        if need_lo.size + need_hi.size == 0:
            continue
        eng.set_zsrc(cfg.cosmo, z_src)
        if clf is not None:
            clf.set_zsrc(cfg.cosmo, z_src)

        def kap3(ra, dec):
            kg, g1g, g2g = eng.kappa_shear_gal(ra, dec, r_out)
            if clf is not None:
                kc, g1c, g2c = clf.kappa_shear_sum(ra, dec)
            else:
                kc = g1c = g2c = 0.0
            return kg, kc, g1g + g1c, g2g + g2c

        kr = np.array([kap3(a, d) for a, d in zip(rra, rdec)])
        k_rand = kr[:, 0] + kr[:, 1]
        zp_c[k] = k_rand.mean()
        zp_g_c[k] = kr[:, 0].mean()
        zp_cl_c[k] = kr[:, 1].mean()
        zp_dmu_c[k] = dmu_exact(k_rand,
                                np.hypot(kr[:, 2], kr[:, 3])).mean()
        sig_c[k] = 0.5 * np.subtract(*np.percentile(k_rand, [84, 16]))
        for i in np.concatenate([need_lo, need_hi]):
            col = 0 if k_lo[i] == k else 1
            kg, kc, g1, g2 = kap3(sn.HOST_RA.iloc[i], sn.HOST_DEC.iloc[i])
            kap_g[i, col] = kg
            kap_c[i, col] = kc
            gam[i, col] = np.hypot(g1, g2)
            dmu[i, col] = float(dmu_exact(kg + kc, gam[i, col]))

    def interp(arr, i, t):
        return (1 - t) * arr[i, 0] + t * (arr[i, 1] if t > 0 else 0.0)

    # deep-NIR availability per galaxy (frac_nir bookkeeping)
    has_nir = np.zeros(len(df), dtype=bool)
    for b in ("y", "j", "h", "ks"):
        if f"mag_{b}" in df:
            has_nir |= np.isfinite(df[f"mag_{b}"].to_numpy(float))

    for i in range(n_sn):
        row_sn = sn.iloc[i]
        t = t_int[i]; lo, hi = k_lo[i], k_hi[i]
        kg = interp(kap_g, i, t); kc = interp(kap_c, i, t)
        dmu_raw = interp(dmu, i, t)
        zp = (1 - t) * zp_c[lo] + t * (zp_c[hi] if t > 0 else 0.0)
        zp_g = (1 - t) * zp_g_c[lo] + t * (zp_g_c[hi] if t > 0 else 0.0)
        zp_cl = (1 - t) * zp_cl_c[lo] + t * (zp_cl_c[hi] if t > 0 else 0.0)
        zp_dmu = (1 - t) * zp_dmu_c[lo] + t * (zp_dmu_c[hi]
                                               if t > 0 else 0.0)
        sig = (1 - t) * sig_c[lo] + t * (sig_c[hi] if t > 0 else 0.0)
        afrac, _ = catalog.area_fraction(cfg, df, row_sn.HOST_RA,
                                         row_sn.HOST_DEC)
        cl_dz = np.nan
        if len(cl):
            sepc = angular_sep_arcsec(row_sn.HOST_RA, row_sn.HOST_DEC,
                                      cl.ra.to_numpy(), cl.dec.to_numpy())
            nearc = sepc < 120.0
            if nearc.any():
                cl_dz = float(np.min(np.abs(cl.z.to_numpy()[nearc]
                                            - row_sn.zHD))
                              / (1.0 + row_sn.zHD))
        results.append({
            "CID": row_sn.CID, "sample": row_sn["sample"],
            "row_index": int(row_sn.get("row_index", -1)),
            "region": int(row.region),
            "zHD": row_sn.zHD, "MU": row_sn.MU, "MUERR": row_sn.MUERR,
            "PROBIA": row_sn.PROBIA,
            "HOST_RA": row_sn.HOST_RA, "HOST_DEC": row_sn.HOST_DEC,
            "HOST_LOGMASS": row_sn.HOST_LOGMASS,
            "clipped": bool(row_sn.clipped),
            "cluster_targeted": bool(row_sn.cluster_targeted),
            "cluster_tier_ok": cl_ok,
            "kappa_raw": kg + kc, "kappa_ext": kg + kc - zp,
            "kappa_gal_ext": kg - zp_g, "kappa_cl_ext": kc - zp_cl,
            "gamma": gam[i, 0 if t < 0.5 else 1],
            "dmu_pred": dmu_raw - zp_dmu,
            "zbin": ZSRC_CENTERS[lo if t < 0.5 else hi],
            "rand_mean": zp, "rand_sig": sig, "n_rand_ok": int(rra.size),
            "area_frac": afrac, "area_flag": afrac < AREA_FLAG_MIN,
            "frac_nir": float(has_nir[
                angular_sep_arcsec(row_sn.HOST_RA, row_sn.HOST_DEC,
                                   df.ra.to_numpy(), df.dec.to_numpy())
                < r_out].mean() if len(df) else 0.0)})


def main():
    global OUTDIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default=str(HD_CSV),
                    help="input Hubble-diagram CSV (union3_prep/"
                         "pantheon_prep schema)")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the N largest regions (smoke test)")
    ap.add_argument("--mstar-method", default="nir1um_fsf",
                    help="stellar-mass estimator (nir_direct_fsf attaches "
                         "deep-NIR photometry from data_nir/)")
    ap.add_argument("--regions", default="",
                    help="comma-separated region ids to process "
                         "(default: all)")
    args = ap.parse_args()
    MSTAR_METHOD[0] = args.mstar_method
    OUTDIR = Path(args.outdir)
    stem = Path(args.hd).stem.replace("_hd", "")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20130901)
    hd = pd.read_csv(args.hd)
    if "row_index" not in hd:
        hd["row_index"] = np.arange(len(hd))
    hd = hd.drop_duplicates("CID", keep="first")
    hd = hd[(hd.zHD > Z_MIN) & (hd.zHD < Z_SRC_MAX - 0.02)].reset_index(
        drop=True)
    n_deep = int((pd.read_csv(HD_CSV).zHD >= Z_SRC_MAX - 0.02).sum())
    log(f"{len(hd)} SNe in ({Z_MIN}, {Z_SRC_MAX - 0.02}); "
        f"{n_deep} beyond (deep-tier, excluded)")

    labels, reg = build_regions(hd.HOST_RA.to_numpy(),
                                hd.HOST_DEC.to_numpy(),
                                hd.zHD.to_numpy(), link_deg=0.5,
                                min_n=2, z_min_singleton=0.2)
    hd["region"] = labels
    proc = reg[reg.process]
    skip = reg[~reg.process]
    lost = hd[hd.region.isin(skip.region)]
    log(f"{len(proc)} regions to process; skipping {len(skip)} low-z "
        f"singletons ({len(lost)} SNe, forecast-weight loss "
        f"{(lost.zHD**2).sum() / (hd.zHD**2).sum() * 100:.1f}% proxy)")
    reg.to_csv(OUTDIR / "regions.csv", index=False)

    cosmo = make_cfg((0.0, 0.0), 1.0, 200).cosmo
    hm = HaloModel(HaloModelConfig(), cosmo, Z_SRC_MAX)
    est = make_estimator(args.mstar_method, cosmo)
    log("shared HaloModel + estimator built")

    proc_iter = proc.sort_values("n_sn", ascending=False)
    if args.regions:
        ids = {int(x) for x in args.regions.split(",")}
        proc_iter = proc_iter[proc_iter.region.isin(ids)]
        log(f"--regions filter: {len(proc_iter)} regions selected")
    if args.limit:
        proc_iter = proc_iter.head(args.limit)

    results, failures = [], []
    for n_done, (_, row) in enumerate(proc_iter.iterrows()):
        sn = hd[hd.region == row.region].reset_index(drop=True)
        try:
            run_region(row, sn, hm, est, rng, results, failures)
        except Exception as exc:
            failures.append({"region": int(row.region), "n_sn": len(sn),
                             "error": str(exc)[:200]})
            log(f"  region {row.region} FAILED: {str(exc)[:120]}")
        if (n_done + 1) % 10 == 0:
            log(f"{n_done + 1}/{len(proc)} regions "
                f"({len(results)} SNe done)")

    res = pd.DataFrame(results)
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=OM_FID)
    res["hr"] = res.MU - cosmo.distmod(res.zHD.to_numpy()).value
    w = 1.0 / res.MUERR**2
    res["hr"] -= np.average(res.hr, weights=w)
    for zb, idx in res.groupby("zbin").groups.items():
        ww = 1.0 / res.loc[idx, "MUERR"] ** 2
        res.loc[idx, "hr"] -= np.average(res.loc[idx, "hr"], weights=ww)
    kappa_csv = OUTDIR / f"{stem}_kappa.csv"
    res.to_csv(kappa_csv, index=False)
    log(f"saved {kappa_csv} ({len(res)} SNe; "
        f"{len(failures)} regions failed)")

    # ---- fits (exclude clipped + cluster-targeted) ----------------------
    good = res[~res.clipped & ~res.cluster_targeted]
    summary = {"n_all": len(res), "n_good": len(good),
               "n_deep_excluded": n_deep, "failures": failures}

    def fit(d, label):
        b, e = bootstrap_slope(d.kappa_ext.to_numpy(), d.hr.to_numpy(),
                               d.MUERR.to_numpy(), rng)
        log(f"{label:30s} N={len(d):4d} slope={b:+.2f}+-{e:.2f} "
            f"({abs(b)/e:.1f} sig) sig_k={d.kappa_ext.std():.4f}")
        return [b, e, len(d)]

    log("=" * 64)
    summary["slope_good"] = fit(good, f"{stem} (field SNe)")
    for s in good["sample"].value_counts().head(6).index:
        d = good[good["sample"] == s]
        if len(d) > 30:
            summary[f"slope_{s}"] = fit(d, f"  {s}")
    b_x, e_x = bootstrap_slope(good.dmu_pred.to_numpy(),
                               good.hr.to_numpy(),
                               good.MUERR.to_numpy(), rng)
    summary["slope_dmu_exact"] = [b_x, e_x]
    log(f"exact-dmu fit: A = {b_x:+.3f} +- {e_x:.3f}")
    (bg, eg), (bc, ec) = two_component_slopes(
        good.kappa_gal_ext.to_numpy(), good.kappa_cl_ext.to_numpy(),
        good.hr.to_numpy(), good.MUERR.to_numpy(), rng)
    summary["two_component"] = {"A_gal": [bg / SLOPE_TH,
                                          eg / abs(SLOPE_TH)],
                                "A_cl": [bc / SLOPE_TH,
                                         ec / abs(SLOPE_TH)]}
    log(f"two-component: A_gal={bg/SLOPE_TH:.2f}+-{eg/abs(SLOPE_TH):.2f} "
        f"A_cl={bc/SLOPE_TH:.2f}+-{ec/abs(SLOPE_TH):.2f}")

    # permutation null within z bins
    x, y, sig = (good.kappa_ext.to_numpy(), good.hr.to_numpy(),
                 good.MUERR.to_numpy())
    zb = good.zbin.to_numpy()
    idx_by = [np.flatnonzero(zb == u) for u in np.unique(zb)]
    yp = y.copy()
    perm = np.empty(2000)
    for k in range(2000):
        for idx in idx_by:
            yp[idx] = y[rng.permutation(idx)]
        perm[k] = np.polyfit(x, yp, 1, w=1.0 / sig)[0]
    b0 = summary["slope_good"][0]
    summary["permutation"] = {
        "p_one_sided": max(float(np.mean(perm <= b0)), 1 / 2000),
        "null_sigma": float(perm.std()),
        "z_equiv": float(abs(b0 - perm.mean()) / perm.std())}
    log(f"permutation: p = {summary['permutation']['p_one_sided']:.2e} "
        f"(z = {summary['permutation']['z_equiv']:.2f})")

    # See-Change validation: cluster-targeted kappa should be elevated
    sc = res[res.cluster_targeted & ~res.clipped]
    if len(sc):
        summary["see_change"] = {
            "n": len(sc),
            "mean_kappa_ext": float(sc.kappa_ext.mean()),
            "field_mean": float(good.kappa_ext.mean()),
            "field_p84": float(np.percentile(good.kappa_ext, 84))}
        log(f"See-Change: <kappa_ext> = {sc.kappa_ext.mean():+.4f} "
            f"(field mean {good.kappa_ext.mean():+.4f}, "
            f"p84 {np.percentile(good.kappa_ext, 84):+.4f})")

    # cross-survey check: same sightlines in the DES-SN5YR catalog
    des = pd.read_csv("output/des_full/des_all_kappa.csv")
    pairs = []
    for _, r_ in good[good["sample"].str.startswith("DES")].iterrows():
        sep = angular_sep_arcsec(r_.HOST_RA, r_.HOST_DEC,
                                 des.HOST_RA.to_numpy(),
                                 des.HOST_DEC.to_numpy())
        j = int(np.argmin(sep))
        if sep[j] < 1.5:
            pairs.append((r_.kappa_ext, des.kappa_ext.iloc[j]))
    if len(pairs) > 10:
        pa = np.array(pairs)
        r_corr = float(np.corrcoef(pa[:, 0], pa[:, 1])[0, 1])
        summary["cross_survey_check"] = {"n_matched": len(pairs),
                                         "corr": r_corr}
        log(f"cross-survey kappa check: {len(pairs)} matched DES3 "
            f"sightlines, corr = {r_corr:.3f}")

    # joint DES + Union3 (drop DES3 overlap + See-Change from Union3 side)
    desg = des[des.PROBIA > 0.9]
    u3j = good[~good["sample"].str.startswith("DES")]
    xj = np.concatenate([desg.kappa_ext, u3j.kappa_ext])
    yj = np.concatenate([desg.hr, u3j.hr])
    sj = np.concatenate([desg.MUERR, u3j.MUERR])
    bj, ej = bootstrap_slope(xj, yj, sj, rng)
    summary["joint_with_des"] = {
        "slope": [bj, ej], "n": len(xj),
        "A": [bj / SLOPE_TH, ej / abs(SLOPE_TH)]}
    log(f"JOINT DES+Union3: slope {bj:+.2f}+-{ej:.2f} "
        f"({abs(bj)/ej:.1f} sig) A = {bj/SLOPE_TH:.2f}+-"
        f"{ej/abs(SLOPE_TH):.2f} (N={len(xj)})")
    summary["n_dmu_clipped"] = N_CLIP[0]
    (OUTDIR / "fit_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    log(f"saved {OUTDIR/'fit_summary.json'}")


if __name__ == "__main__":
    main()
