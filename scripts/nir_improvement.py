#!/usr/bin/env python
"""Deep-NIR mass upgrade, Phase 4: quantify the improvement.

Stages:
  --stage noise : per-SN classical/Berkson noise decompositions for the
      NIR-field subsamples (DES X/C/E + Union3/Pantheon+ NIR regions),
      with the NirDirectFSF estimator and a REDUCED per-galaxy M*
      sigma for NIR-interpolated galaxies (sigma_nir^2 = sigma_cfg^2 -
      removed^2 from output/nir_video/mstar_validation.json, floored at
      0.08 dex). Galaxies without deep NIR keep the config scatter.
      Output: output/nir_video/latent_noise_nir.csv
  --stage fit : everything downstream (fast) --
      (i) per-sightline kappa_new vs kappa_old (corr, medians, frac_nir);
      (ii) WLS slope + exact-dmu amplitude, old vs new, on the matched
           subsample with hr re-demeaned identically for both;
      (iii) latent EIV amplitude old vs new on the deduplicated joint
           NIR subsample (old = tracked catalogs + tracked noise csvs;
           new = NIR catalogs + latent_noise_nir.csv), the shrinkage
           weight <w> being the attenuation statement;
      (iv) projection of the joint-fit error if the NIR gain applied
           program-wide (the Euclid statement).
      Output: output/nir_video/improvement.json

Run:  .venv/bin/python scripts/nir_improvement.py --stage noise
      .venv/bin/python scripts/nir_improvement.py --stage fit
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa import catalog, clusters as clu, montecarlo
from snkappa.datalab import TapClient
from snkappa.fitting import bootstrap_slope
from snkappa.halos import HaloModel
from snkappa.kappa import KappaEngine, angular_sep_arcsec
from snkappa.latent import fit_amplitude
from snkappa.nir import attach_nir
from snkappa.stellar import make_estimator
import des_full
import union3_full as uf
from latent_fit import rvar, load_merged, prep_arrays

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


OUTDIR = Path("output/nir_video")
NOISE_CSV = OUTDIR / "latent_noise_nir.csv"
NEW_KAPPA = {
    "des": OUTDIR / "des_all_kappa_nir.csv",
    "union3": OUTDIR / "union3/union3_kappa.csv",
    "pantheon": OUTDIR / "pantheon/pantheon_kappa.csv",
}
OLD_KAPPA = {
    "des": Path("output/des_full/des_all_kappa.csv"),
    "union3": Path("output/union3/union3_kappa.csv"),
    "pantheon": Path("output/pantheon/pantheon_kappa.csv"),
}
OLD_NOISE = {
    "des": Path("output/des_full/latent_noise.csv"),
    "union3": Path("output/union3/latent_noise.csv"),
    "pantheon": Path("output/pantheon/latent_noise.csv"),
}
SLOPE_TH = -5.0 / np.log(10.0)
AFID = 0.79


def sigma_nir():
    """Reduced per-galaxy M* sigma for NIR-interpolated galaxies."""
    v = json.loads((OUTDIR / "mstar_validation.json").read_text())
    quad = v["gate"]["removed_scatter_quadrature_dex"]
    sig_cfg = 0.15          # HaloModelConfig.mstar_scatter_dex default
    return float(np.sqrt(max(sig_cfg ** 2 - quad ** 2, 0.08 ** 2))), quad


def _noise_rows(cfg0, df, members, cl, hm, est, logms_err, gsn, rng, n_mc,
                rows, tag):
    r_ap = des_full.APERTURE_ARCMIN * 60.0
    for j, (_, sn) in enumerate(gsn.iterrows()):
        z_src = float(sn.zHD)
        theta = angular_sep_arcsec(sn.HOST_RA, sn.HOST_DEC,
                                   df.ra.to_numpy(), df.dec.to_numpy())
        in_ap = theta < r_ap + 30.0
        df_ap = df[in_ap].reset_index(drop=True)
        memb_ap = members[in_ap]
        err_ap = logms_err[in_ap]

        cfg = copy.deepcopy(cfg0)
        cfg.source.ra_src = float(sn.HOST_RA)
        cfg.source.dec_src = float(sn.HOST_DEC)
        cfg.source.z_src = z_src
        cfg.montecarlo.n_mc = n_mc
        eng = KappaEngine(cfg, cfg.cosmo, hm, est, df_ap,
                          logms_err=err_ap)

        cfg_cl = copy.deepcopy(cfg)
        cfg_cl.halo_model.smhm_scatter_dex = 1e-4
        cfg_cl.halo_model.c_scatter_dex = 1e-4
        slo, shi = eng.phot_slo, eng.phot_shi
        eng.phot_slo = np.full_like(slo, 1e-4)
        eng.phot_shi = np.full_like(shi, 1e-4)
        draws_cl = montecarlo.mc_kappa_raw(cfg_cl, eng, rng, memb_ap)
        eng.phot_slo, eng.phot_shi = slo, shi
        draws_tot = montecarlo.mc_kappa_raw(cfg, eng, rng, memb_ap)

        cl_fg = cl[cl.z.to_numpy() < z_src - 0.02] if len(cl) else cl
        nmc = min(n_mc, 48)
        if len(cl_fg):
            ck = clu.ClusterKappa(cfg, hm, cfg.cosmo, cl_fg)
            n_fg = len(ck.df)
            cl_cl = np.array([ck.kappa_sum(
                sn.HOST_RA, sn.HOST_DEC,
                dlogm=rng.normal(0.0, cfg.clusters.mass_scatter_dex,
                                 n_fg)) for _ in range(nmc)]) \
                if n_fg else np.zeros(nmc)
            cl_tot = ck.mc_kappa_sum(sn.HOST_RA, sn.HOST_DEC, rng, nmc) \
                if n_fg else np.zeros(nmc)
        else:
            cl_cl = cl_tot = np.zeros(nmc)
        rows.append({
            "CID": sn.CID, "survey": tag, "zHD": z_src,
            "rvar_gal_cl": rvar(draws_cl), "rvar_gal_tot": rvar(draws_tot),
            "rvar_cl_cl": rvar(cl_cl), "rvar_cl_tot": rvar(cl_tot),
            "var_zp": (sn.rand_sig ** 2) / max(int(sn.n_rand_ok), 1)})
        if (j + 1) % 50 == 0:
            log(f"  {tag}: {j + 1}/{len(gsn)} ({len(rows)} total)")


def nir_logms_err(df, sig_nir):
    """Per-galaxy M* sigma: reduced where deep-NIR photometry exists."""
    has = np.zeros(len(df), dtype=bool)
    for b in ("y", "j", "h", "ks"):
        if f"mag_{b}" in df:
            has |= np.isfinite(df[f"mag_{b}"].to_numpy(float))
    return np.where(has, sig_nir, np.nan)   # NaN -> config default


def stage_noise(args):
    sig_nir, quad = sigma_nir()
    log(f"sigma_nir = {sig_nir:.3f} dex (removed {quad:.3f} in quadrature "
        f"from 0.15 config default)")
    rng = np.random.default_rng(27182)
    rows = []

    # --- DES X/C/E ------------------------------------------------------
    res = pd.read_csv(NEW_KAPPA["des"])
    for gname in ("X", "C", "E"):
        gsn = res[res.GROUP == gname]
        if args.limit:
            gsn = gsn.iloc[:args.limit]
        fields, center = des_full.FIELD_GROUPS[gname]
        rad = angular_sep_arcsec(center[0], center[1],
                                 res[res.GROUP == gname].HOST_RA.to_numpy(),
                                 res[res.GROUP == gname].HOST_DEC.to_numpy()
                                 ).max() / 3600.0

        class A:
            smhm_inverse = "posterior"; logmh_max = 13.8; n_rand = 500
        cfg0 = des_full.make_cfg(A, center, rad + 0.25)
        tap = TapClient(cfg0.data.tap_url, cfg0.data.cache_dir)
        df = catalog.clean_and_merge(cfg0,
                                     *catalog.fetch_regional(cfg0, tap))
        df = attach_nir(df)
        cl = clu.fetch_clusters(cfg0,
                                lambda u: TapClient(u, cfg0.data.cache_dir))
        hm = HaloModel(cfg0.halo_model, cfg0.cosmo, 1.15)
        members = clu.assign_members(cfg0, df, cl, hm)
        est = make_estimator("nir_direct_fsf", cfg0.cosmo)
        err = nir_logms_err(df, sig_nir)
        log(f"DES {gname}: catalog {len(df)} "
            f"({int(np.isfinite(err).sum())} NIR), {len(gsn)} SNe")
        _noise_rows(cfg0, df, members, cl, hm, est, err, gsn, rng,
                    args.n_mc, rows, f"des:{gname}")

    # --- Union3 / Pantheon+ NIR regions ---------------------------------
    for survey in ("union3", "pantheon"):
        res = pd.read_csv(NEW_KAPPA[survey])
        reg = pd.read_csv(OUTDIR / survey / "regions.csv"
                          ).set_index("region")
        cosmo = uf.make_cfg((0.0, 0.0), 1.0, 200).cosmo
        hm = HaloModel(uf.HaloModelConfig(), cosmo, uf.Z_SRC_MAX)
        est = make_estimator("nir_direct_fsf", cosmo)
        for rid, grp in res.groupby("region"):
            if args.limit:
                grp = grp.iloc[:args.limit]
            rrow = reg.loc[rid]
            cfg0 = uf.make_cfg((rrow.ra, rrow.dec),
                               max(rrow.radius_deg + 0.25, 0.45), 200)
            tap = TapClient(cfg0.data.tap_url, cfg0.data.cache_dir)
            df = catalog.clean_and_merge(
                cfg0, *catalog.fetch_regional(cfg0, tap))
            df = attach_nir(df)
            cl = (uf.local_clusters((rrow.ra, rrow.dec),
                                    catalog.region_radius_deg(cfg0))
                  if uf.WH_LOCAL.exists() else
                  pd.DataFrame(columns=["name", "ra", "dec", "z", "m200"]))
            members = (clu.assign_members(cfg0, df, cl, hm) if len(cl)
                       else np.zeros(len(df), dtype=bool))
            err = nir_logms_err(df, sig_nir)
            _noise_rows(cfg0, df, members, cl, hm, est, err, grp, rng,
                        args.n_mc, rows, f"{survey}:{rid}")
        log(f"{survey}: done ({len(rows)} total)")

    pd.DataFrame(rows).to_csv(NOISE_CSV, index=False)
    log(f"saved {NOISE_CSV} ({len(rows)} SNe)")


# ------------------------------------------------------------------ fit
def redemean(d, om):
    """hr from MU with global + per-zbin weighted demean (driver logic),
    applied identically to any (sub)sample."""
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70.0, Om0=om)
    hr = d.MU.to_numpy() - cosmo.distmod(d.zHD.to_numpy()).value
    w = 1.0 / d.MUERR.to_numpy() ** 2
    hr = hr - np.average(hr, weights=w)
    zb = d.zbin.to_numpy()
    for u in np.unique(zb):
        s = zb == u
        hr[s] -= np.average(hr[s], weights=w[s])
    return hr


def good_mask(d):
    m = d.PROBIA.to_numpy() > 0.9
    if "clipped" in d:
        m &= ~d.clipped.to_numpy().astype(bool)
    if "cluster_targeted" in d:
        m &= ~d.cluster_targeted.to_numpy().astype(bool)
    return m


def stage_fit(args):
    rng = np.random.default_rng(31415)
    out = {}
    sig_nir, quad = sigma_nir()
    out["sigma_nir_dex"] = sig_nir

    oms = {"des": 0.352, "union3": 0.356, "pantheon": 0.356}
    pairs = {}
    for s in ("des", "union3", "pantheon"):
        new = pd.read_csv(NEW_KAPPA[s])
        old = pd.read_csv(OLD_KAPPA[s])
        m = new.merge(old, on="CID", suffixes=("_new", "_old"))
        pairs[s] = m
        ok = np.isfinite(m.kappa_ext_new) & np.isfinite(m.kappa_ext_old)
        dk = (m.kappa_ext_new - m.kappa_ext_old)[ok]
        out[f"{s}_kappa"] = {
            "n": int(ok.sum()),
            "corr": float(np.corrcoef(m.kappa_ext_new[ok],
                                      m.kappa_ext_old[ok])[0, 1]),
            "median_abs_dkappa": float(np.median(np.abs(dk))),
            "std_old": float(m.kappa_ext_old[ok].std()),
            "std_new": float(m.kappa_ext_new[ok].std()),
            "mean_frac_nir": float(m.frac_nir.mean())
            if "frac_nir" in m else None}
        # implied excess-variance removal: regression coefficient of
        # kappa_new on kappa_old on fully-covered sightlines. If
        # new = true + small noise and old = true + extra (mass-error)
        # noise, cov(new, old) ~ Var(true), so b ~ Var(true)/Var(old) --
        # the attenuation factor the OLD predictions suffered relative
        # to the new ones. Correlated-across-galaxies mass errors are
        # captured here but invisible to the independent-jitter MC
        # (the same physics that makes lambda_mock 0.68 < analytic 0.89).
        if "frac_nir" in m:
            f = ok & (m.frac_nir > 0.8)
            if f.sum() > 100:
                for tag, cn, co in (("total", "kappa_ext_new",
                                     "kappa_ext_old"),
                                    ("gal", "kappa_gal_ext_new",
                                     "kappa_gal_ext_old")):
                    b = (np.cov(m[cn][f], m[co][f])[0, 1]
                         / np.var(m[co][f]))
                    out[f"{s}_kappa"][f"b_new_on_old_{tag}"] = float(b)
        log(f"{s}: N={ok.sum()} corr={out[f'{s}_kappa']['corr']:.4f} "
            f"std {out[f'{s}_kappa']['std_old']:.4f}->"
            f"{out[f'{s}_kappa']['std_new']:.4f} "
            f"frac_nir={out[f'{s}_kappa']['mean_frac_nir']}")

    # --- WLS slopes old vs new, hr re-demeaned identically ---------------
    for s, m in pairs.items():
        g = m[good_mask(m.rename(columns={
            "PROBIA_new": "PROBIA", "clipped_new": "clipped",
            "cluster_targeted_new": "cluster_targeted"}))].copy()
        res_o = g.rename(columns={"MU_new": "MU", "MUERR_new": "MUERR",
                                  "zHD_new": "zHD",
                                  "zbin_new": "zbin"})
        hr = redemean(res_o, oms[s])
        muerr = g.MUERR_new.to_numpy()
        b_o, e_o = bootstrap_slope(g.kappa_ext_old.to_numpy(), hr,
                                   muerr, rng)
        b_n, e_n = bootstrap_slope(g.kappa_ext_new.to_numpy(), hr,
                                   muerr, rng)
        a_o, ae_o = bootstrap_slope(g.dmu_pred_old.to_numpy(), hr,
                                    muerr, rng)
        a_n, ae_n = bootstrap_slope(g.dmu_pred_new.to_numpy(), hr,
                                    muerr, rng)
        out[f"{s}_wls"] = {
            "n": len(g),
            "slope_old": [b_o, e_o], "slope_new": [b_n, e_n],
            "A_old": [b_o / SLOPE_TH, e_o / abs(SLOPE_TH)],
            "A_new": [b_n / SLOPE_TH, e_n / abs(SLOPE_TH)],
            "A_exact_old": [a_o, ae_o], "A_exact_new": [a_n, ae_n]}
        log(f"{s} WLS: old {b_o:+.2f}+-{e_o:.2f} | new {b_n:+.2f}+-{e_n:.2f}"
            f"  (A {b_o/SLOPE_TH:.2f} -> {b_n/SLOPE_TH:.2f})")

    # --- latent EIV old vs new on the joint deduplicated subsample --------
    from joint_fit import match_pairs
    nnoise = pd.read_csv(NOISE_CSV)

    def latent_frames(which):
        frames = []
        for s, tag in (("des", "DES"), ("union3", "Union3"),
                       ("pantheon", "PantheonP")):
            if which == "old":
                d = load_merged(OLD_KAPPA[s], noise_csv=OLD_NOISE[s])
                keep = set(pairs[s].CID)
                d = d[d.CID.isin(keep)]
            else:
                res = pd.read_csv(NEW_KAPPA[s])
                res["CID"] = res.CID.astype(str)
                nn = nnoise[nnoise.survey.str.startswith(
                    s if s != "des" else "des")][[
                        "CID", "rvar_gal_cl", "rvar_gal_tot",
                        "rvar_cl_cl", "rvar_cl_tot", "var_zp"]].copy()
                nn["CID"] = nn.CID.astype(str)
                d = res.merge(nn, on="CID")
                d = d[d.PROBIA > 0.9]
                if "clipped" in d:
                    d = d[~d.clipped.astype(bool)]
                if "cluster_targeted" in d:
                    d = d[~d.cluster_targeted.astype(bool)]
                d = d.reset_index(drop=True)
            x, y, zbin, s2c, s2b, sig2y, _ = prep_arrays(d, afid=AFID)
            frames.append(pd.DataFrame({
                "x": x, "y": y, "s2c": s2c, "s2b": s2b, "sig2y": sig2y,
                "zbin": [f"{tag}|{z}" for z in zbin],
                "HOST_RA": d.HOST_RA, "HOST_DEC": d.HOST_DEC,
                "zHD": d.zHD, "MUERR": d.MUERR, "compilation": tag}))
        return pd.concat(frames, ignore_index=True)

    def dedup(d):
        n = len(d)
        parent = np.arange(n)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for ca, cb in (("DES", "Union3"), ("Union3", "PantheonP"),
                       ("DES", "PantheonP")):
            for i, j in match_pairs(d, ca, cb):
                parent[find(i)] = find(j)
        comp = np.array([find(i) for i in range(n)])
        sig = d.MUERR.to_numpy()
        rep = [m[np.argmin(sig[m])] for m in
               (np.flatnonzero(comp == u) for u in np.unique(comp))]
        return d.iloc[rep].reset_index(drop=True)

    for which in ("old", "new"):
        r = dedup(latent_frames(which))
        fit = fit_amplitude(r.x.to_numpy(), r.s2c.to_numpy(),
                            r.zbin.to_numpy(), r.y.to_numpy(),
                            r.sig2y.to_numpy(), rng=rng, n_boot=200,
                            s2_berk=r.s2b.to_numpy())
        out[f"latent_{which}"] = {
            "n_unique": len(r), "A": fit["A"], "A_err": fit["A_err"],
            "A_err_post": fit["A_err_post"],
            "mean_reliability": fit["mean_reliability"],
            "mean_s2c": float(r.s2c.mean())}
        log(f"latent {which}: A = {fit['A']:.3f} +- {fit['A_err']:.3f} "
            f"(<w> = {fit['mean_reliability']:.3f}, "
            f"n = {len(r)})")

    # --- projection: the gain is ACCURACY, not precision ------------------
    # N_SN is unchanged, so the statistical error on A does not shrink.
    # What the NIR masses buy: (i) the WLS/exact amplitudes rise toward
    # unity in all three samples (less unmodeled attenuation to correct);
    # (ii) the classical (shrinkage-driving) noise drops (s2c ratio
    # below), so the latent A needs less prior-driven correction; (iii)
    # the old predictions' excess variance (b_new_on_old < 1) was
    # invisible to the independent-jitter MC -- with NIR masses less of
    # the attenuation budget rides on the mock-calibrated lambda.
    b_ratios = [out[f"{s}_kappa"].get("b_new_on_old_total")
                for s in ("des", "union3", "pantheon")]
    b_ratios = [b for b in b_ratios if b is not None]
    a_shift = {s: [out[f"{s}_wls"]["A_exact_old"][0],
                   out[f"{s}_wls"]["A_exact_new"][0]]
               for s in ("des", "union3", "pantheon")}
    out["projection"] = {
        "exact_A_old_new_per_survey": a_shift,
        "mean_b_new_on_old_infootprint": float(np.mean(b_ratios))
        if b_ratios else None,
        "s2c_ratio_new_over_old": float(
            out["latent_new"]["mean_s2c"] / out["latent_old"]["mean_s2c"]),
        "note": ("accuracy gain: amplitudes shift toward unity and the "
                 "old predictions' excess (mass-error) variance is "
                 "removed; statistical A errors are SN-scatter limited "
                 "and unchanged. Program-wide deep NIR (Euclid Y/J/H "
                 "over the full footprints) applies the same per-galaxy "
                 "gain to the remaining ~half of the joint signal.")}
    log(f"projection: mean in-footprint b_new_on_old = "
        f"{out['projection']['mean_b_new_on_old_infootprint']}; "
        f"s2c ratio {out['projection']['s2c_ratio_new_over_old']:.2f}")

    (OUTDIR / "improvement.json").write_text(
        json.dumps(out, indent=2, default=float))
    log(f"saved {OUTDIR/'improvement.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("noise", "fit"))
    ap.add_argument("--n-mc", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "noise":
        stage_noise(args)
    else:
        stage_fit(args)


if __name__ == "__main__":
    main()
