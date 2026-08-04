#!/usr/bin/env python
"""Assemble the released FrankenBlast mass catalog (close-out P3).

Merges every fitting tier into one row per target:
  - zfix SBI sweep + mop-up  (engine = sbi_zfix)
  - direct prospector tier-2 (engine = prospector), for the galaxies
    the emulator could not fit
with per-galaxy sigma_z (fb_sigz.py) and
  sigma_total = posterior halfwidth  (+)  sigma_M(photz)
where sigma_M(photz) = |dlogM*/dz|(z) * sig_z using the binned response
curve measured by fb_photoz_sensitivity.py (galaxies inside the refit
subsample get their directly measured value).

Output: output/nir_video/fb_masses.csv (tracked; one row per fitted
target, no duplicates).

Run: .venv/bin/python scripts/fb_masses.py
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fb_photoz_sensitivity import rows  # noqa: E402  shared robust reader

OUT = Path("output/nir_video/fb_masses.csv")


def collect_prospector_npz(pattern):
    """Tier-2 masses straight from the saved dynesty posteriors -- the
    prospector summary CSVs have a third column layout and the same
    append-under-wrong-header failure mode, so the .npz posteriors are
    the reliable record."""
    rec = {}
    for f in glob.glob(pattern):
        try:
            d = np.load(f, allow_pickle=True)
            labels = list(d["labels"])
            j = labels.index("logmass")
            s = np.asarray(d["samples"])[:, j]
            logwt = np.asarray(d["logwt"])
            w = np.exp(logwt - logwt.max())
            w /= w.sum()
            i = np.argsort(s)
            c = np.cumsum(w[i]); c /= c[-1]
            p16, p50, p84 = np.interp([0.16, 0.5, 0.84], c, s[i])
            ls = Path(f).stem.replace("prospector_ls", "")
            rec[ls] = {"engine": "prospector",
                       "logm_p16": float(p16), "logm_p50": float(p50),
                       "logm_p84": float(p84), "runtime_s": np.nan}
        except Exception:
            continue
    return rec


def collect(pattern, engine):
    rec = {}
    for f in glob.glob(pattern):
        for r in rows(f):
            if r.get("status") != "ok" or not r.get("logmass_p50"):
                continue
            try:
                rec[str(r["ls_id"])] = {
                    "engine": engine,
                    "logm_p16": float(r["logmass_p16"]),
                    "logm_p50": float(r["logmass_p50"]),
                    "logm_p84": float(r["logmass_p84"]),
                    "runtime_s": float(r["runtime_s"])}
            except (ValueError, TypeError):
                pass
    return rec


def main():
    fits = collect("data_nir/summaries_zfix/summary_zfix_*.csv",
                   "sbi_zfix")
    t2 = collect_prospector_npz("data_nir/pros_npz/prospector_ls*.npz")
    tier2_ids = {r["ls_id"] for r in
                 pd.read_csv("data_nir/fb_targets_todo.csv",
                             dtype={"ls_id": str}).to_dict("records")} \
        if Path("data_nir/fb_targets_todo.csv").exists() else set(t2)
    for k, v in t2.items():
        if k in tier2_ids:
            fits.setdefault(k, v)  # tier-2 only where the SBI failed
    print(f"{len(fits)} fitted galaxies "
          f"({sum(1 for v in fits.values() if v['engine']=='prospector')}"
          f" prospector tier-2)")

    sz = pd.read_csv("data_nir/fb_targets_sigz.csv")
    sz["ls_id"] = sz.ls_id.astype(str)
    tg = pd.read_csv("data_nir/fb_targets_all.csv")
    tg["ls_id"] = tg.ls_id.astype(str)
    tg = tg[["ls_id", "ra", "dec", "kappa_max", "kappa_sum", "n_sn",
             "survey_region"]].merge(sz, on="ls_id")

    sens = json.loads(Path("output/nir_video/"
                           "fb_photoz_sensitivity.json").read_text())
    curve = sens["response_curve"]
    edges = [tuple(map(float, k.split("-"))) for k in curve]
    vals = list(curve.values())

    def response(z):
        for (lo, hi), v in zip(edges, vals):
            if lo <= z < hi:
                return v
        return vals[-1] if z >= edges[-1][1] else vals[0]

    # directly measured sigma for the refit subsample
    measured = {}
    pm_sub = Path("data_nir/fb_pm_subsample.csv")
    plus = collect("data_nir/summaries_zfix_pm/summary_pm_plus_*.csv",
                   "x")
    minus = collect("data_nir/summaries_zfix_pm/summary_pm_minus_*.csv",
                    "x")
    if pm_sub.exists():
        for _, r in pd.read_csv(pm_sub).iterrows():
            i = str(r.ls_id)
            if i in plus and i in minus:
                measured[i] = 0.5 * abs(plus[i]["logm_p50"]
                                        - minus[i]["logm_p50"])

    out = []
    for _, r in tg.iterrows():
        i = r.ls_id
        if i not in fits:
            continue
        v = fits[i]
        half = 0.5 * (v["logm_p84"] - v["logm_p16"])
        sig_pz = measured.get(i)
        if sig_pz is None:
            sig_pz = (0.0 if bool(r.z_is_spec)
                      else response(float(r.z_best)) * float(r.sig_z))
        out.append({
            "ls_id": i, "ra": r.ra, "dec": r.dec,
            "z_best": r.z_best, "z_is_spec": r.z_is_spec,
            "sig_z": r.sig_z, "engine": v["engine"],
            "logm_p16": v["logm_p16"], "logm_p50": v["logm_p50"],
            "logm_p84": v["logm_p84"],
            "sig_m_photz": round(sig_pz, 4),
            "sig_m_total": round(float(np.hypot(half, sig_pz)), 4),
            "kappa_max": r.kappa_max, "kappa_sum": r.kappa_sum,
            "n_sn": r.n_sn, "survey_region": r.survey_region})
    df = pd.DataFrame(out).sort_values("kappa_max", ascending=False)
    assert df.ls_id.is_unique
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {OUT}: {len(df)} rows; sig_m_total median "
          f"{df.sig_m_total.median():.3f} dex "
          f"(spec-z rows {df.z_is_spec.astype(bool).sum()})")


if __name__ == "__main__":
    main()
