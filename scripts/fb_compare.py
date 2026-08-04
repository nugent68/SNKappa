#!/usr/bin/env python
"""Compare FrankenBlast 8-band (zfix) masses against the SNKappa
estimators, and decide whether the mass scale needs recalibrating.

Consumes the campaign summaries produced on Perlmutter
(scripts/frankenblast/, mode `run_kappa_zfix.py`) joined to the target
photometry written by scripts/fb_targets.py, and evaluates the same
galaxies with NirDirectFSF (deep-NIR) and Nir1umFSF (legacy z<->W1)
from identical photometry, so the only difference is the estimator.

Outputs output/nir_video/fb_compare.json (tracked):
  - offsets and NMAD, overall and in logM* / z / kappa bins
  - the massive-end statement the campaign was built to test (the
    5-band run had FB +0.42 dex high there)
  - a kappa-weighted offset: what a coherent mass shift would do to
    the predicted convergence, which is what actually matters for the
    lensing amplitude
  - cross-checks against the independent references already measured
    (CIGALE / LePhare, output/nir_video/mstar_validation.json) so the
    FB scale is placed among them rather than trusted blindly

Usage:
  scp the summaries locally first, e.g.
    scp -r perlmutter:/pscratch/sd/n/nugent/lens/summaries_zfix data_nir/
  then
    .venv/bin/python scripts/fb_compare.py [--summaries data_nir/summaries_zfix]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snkappa.stellar import NirDirectFSF, Nir1umFSF

OUT = Path("output/nir_video/fb_compare.json")
TARGETS = Path("data_nir/fb_targets_all.csv")
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.352)
SLOPE_TH = -5.0 / np.log(10.0)


def nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def load_summaries(d):
    frames = []
    for f in glob.glob(f"{d}/summary_zfix_n*_w*.csv"):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        raise SystemExit(f"no summaries found under {d}")
    s = pd.concat(frames, ignore_index=True)
    s = s[s.status == "ok"].copy()
    s["ls_id"] = s.ls_id.astype(str)
    # a galaxy can appear twice if a worker retried; keep the first
    return s.drop_duplicates("ls_id", keep="first")


def mags_from(df):
    def m(f, mw=None):
        f = np.asarray(f, float)
        mw = np.ones_like(f) if mw is None else np.asarray(mw, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(f > 0,
                            22.5 - 2.5 * np.log10(
                                f / np.where(mw > 0, mw, np.nan)),
                            np.nan)
    return {"g": m(df.flux_g, df.mw_transmission_g),
            "r": m(df.flux_r, df.mw_transmission_r),
            "z": m(df.flux_z, df.mw_transmission_z),
            "w1": m(df.flux_w1, df.mw_transmission_w1),
            "j": m(df.flux_j), "h": m(df.flux_h), "ks": m(df.flux_k)}


def block(d, label):
    """Offset stats for one selection."""
    return {"label": label, "n": int(len(d)),
            "median_nirdirect_minus_fb": float(np.median(d.d_new)),
            "nmad_nirdirect": nmad(d.d_new),
            "median_nir1um_minus_fb": float(np.median(d.d_old)),
            "nmad_nir1um": nmad(d.d_old),
            "median_fb_logm": float(np.median(d.fb_logm))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", default="data_nir/summaries_zfix")
    args = ap.parse_args()

    s = load_summaries(args.summaries)
    tg = pd.read_csv(TARGETS)
    tg["ls_id"] = tg.ls_id.astype(str)
    m = s.merge(tg, on="ls_id", suffixes=("_fb", ""))
    m = m.rename(columns={"logmass_p50": "fb_logm",
                          "logmass_p16": "fb_p16",
                          "logmass_p84": "fb_p84"})
    print(f"{len(s)} FB fits, {len(m)} joined to targets")

    mags = mags_from(m)
    z = m.z_best.to_numpy(float)
    m["ours_new"] = NirDirectFSF(COSMO).logmstar(mags, z)
    m["ours_old"] = Nir1umFSF(COSMO).logmstar(
        {k: v for k, v in mags.items() if k in ("g", "r", "z", "w1")}, z)
    m["d_new"] = m.ours_new - m.fb_logm
    m["d_old"] = m.ours_old - m.fb_logm
    m["fb_err"] = 0.5 * (m.fb_p84 - m.fb_p16)
    good = m[np.isfinite(m.d_new) & np.isfinite(m.d_old)
             & (m.fb_logm > 7)].copy()

    out = {"n_fits": int(len(s)), "n_compared": int(len(good)),
           "overall": block(good, "all"),
           "median_fb_posterior_halfwidth": float(
               np.median(good.fb_err))}

    # the headline: massive end (where the 5-band run was +0.42 dex off)
    out["bins_logm"] = [block(good[(good.fb_logm >= lo)
                                   & (good.fb_logm < hi)],
                              f"fb_logM {lo}-{hi}")
                        for lo, hi in ((9.0, 10.0), (10.0, 10.5),
                                       (10.5, 11.0), (11.0, 11.5),
                                       (11.5, 13.0))
                        if ((good.fb_logm >= lo)
                            & (good.fb_logm < hi)).sum() >= 20]
    out["bins_z"] = [block(good[(good.z_best >= lo) & (good.z_best < hi)],
                           f"z {lo}-{hi}")
                     for lo, hi in ((0.05, 0.3), (0.3, 0.5),
                                    (0.5, 0.8), (0.8, 1.2))
                     if ((good.z_best >= lo) & (good.z_best < hi)).sum() >= 20]
    spec = good[good.z_is_spec.astype(bool)]
    if len(spec) >= 20:
        out["spec_z_only"] = block(spec, "DESI spec-z subset")

    # kappa-weighted offset: a coherent mass shift moves the predicted
    # convergence roughly as dlog kappa ~ (dlogM_h/dlogM*) * dlogM*;
    # report the contribution-weighted mean offset as the quantity that
    # propagates into the lensing amplitude.
    w = good.kappa_sum.to_numpy(float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() > 0:
        out["kappa_weighted"] = {
            "mean_offset_nirdirect": float(np.average(good.d_new, weights=w)),
            "mean_offset_nir1um": float(np.average(good.d_old, weights=w)),
            "note": ("kappa_sum-weighted mean of (ours - FB); the "
                     "unweighted median understates the influence of "
                     "the few halos that dominate the prediction")}

    # place the FB scale among the references already measured
    val = Path("output/nir_video/mstar_validation.json")
    if val.exists():
        v = json.loads(val.read_text())
        out["reference_context"] = {
            "nirdirect_minus_cigale_median":
                v["vs_cigale"]["med_new"],
            "nirdirect_minus_lephare_median":
                v["vs_lephare_cosmos"]["med_new"],
            "note": ("if NirDirectFSF sits below FB by a similar amount "
                     "as it sits below CIGALE/LePhare, the offset is a "
                     "zero-point of our estimator, not an FB artifact")}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    o = out["overall"]
    print(f"NirDirectFSF - FB : {o['median_nirdirect_minus_fb']:+.3f} dex "
          f"(NMAD {o['nmad_nirdirect']:.3f}, N={o['n']})")
    print(f"Nir1umFSF    - FB : {o['median_nir1um_minus_fb']:+.3f} dex "
          f"(NMAD {o['nmad_nir1um']:.3f})")
    for b in out["bins_logm"]:
        print(f"  {b['label']:>18s}: NirDirect {b['median_nirdirect_minus_fb']:+.3f} "
              f"(NMAD {b['nmad_nirdirect']:.3f}, n={b['n']})")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
