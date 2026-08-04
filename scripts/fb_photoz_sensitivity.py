#!/usr/bin/env python
"""Photo-z -> mass uncertainty for the FB zfix catalog (close-out P3).

The zfix fits condition on z_best, so their posteriors omit the
photo-z uncertainty. This measures the response empirically from the
+/- sigma_z refits: for each subsample galaxy,

    dlogM*/dz      = (logM_plus - logM_minus) / (2 sig_z)
    sigma_M(photz) = |logM_plus - logM_minus| / 2

and reports the distribution, its redshift dependence (against the
analytic 2 dlog10 D_L/dz luminosity-distance expectation), the
kappa-weighted version, and a binned response curve used by
fb_masses.py to assign sigma_M(photz) to galaxies outside the refit
subsample.

Inputs (scp from Perlmutter first):
    data_nir/summaries_zfix/       main-sweep summaries (central fits)
    data_nir/summaries_zfix_pm/    +/- sigma_z summaries
    data_nir/fb_pm_subsample.csv   subsample (ls_id, z_best, sig_z, kappa_max)

Output: output/nir_video/fb_photoz_sensitivity.json (tracked)

Run: .venv/bin/python scripts/fb_photoz_sensitivity.py
"""

import csv
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

OUT = Path("output/nir_video/fb_photoz_sensitivity.json")
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.352)
HDR = "ls_id,lensed,status,n_filters,runtime_s".split(",")
# full column order of run_lens_catalog summary rows (5 fixed +
# p16/p50/p84 for each CHAIN_KEYS entry)
FULL = HDR + [f"{k}_{p}" for k in
              ("zred", "logmass", "logzsol", "age", "dust_AV", "SFR",
               "log_fagn", "log_agn_tau") for p in ("p16", "p50", "p84")]


def rows(f):
    """Summary reader robust to BOTH file pathologies (see
    scripts/frankenblast/README.md): headerless files from the timeout
    fallback, and 5-column pre-written headers with 29-column data rows
    appended under them (the pm/mop runners)."""
    with open(f) as fh:
        first = fh.readline().rstrip("\n")
        header_fields = first.split(",") if first.startswith("ls_id") else None
        fh.seek(0)
        if header_fields is not None and len(header_fields) >= len(FULL):
            yield from csv.DictReader(fh)
            return
        if header_fields is not None:
            fh.readline()          # skip the short header
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= len(FULL):
                yield dict(zip(FULL, p))
            elif len(p) >= 5:
                yield dict(zip(HDR, p[:5]))


def load_masses(pattern):
    out = {}
    for f in glob.glob(pattern):
        for r in rows(f):
            if r.get("status") == "ok" and r.get("logmass_p50"):
                try:
                    out[str(r["ls_id"])] = (
                        float(r["logmass_p50"]),
                        0.5 * (float(r["logmass_p84"])
                               - float(r["logmass_p16"])))
                except (ValueError, TypeError):
                    pass
    return out


def nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size \
        else float("nan")


def main():
    sub = pd.read_csv("data_nir/fb_pm_subsample.csv")
    sub["ls_id"] = sub.ls_id.astype(str)
    cen = load_masses("data_nir/summaries_zfix/summary_zfix_*.csv")
    plus = load_masses("data_nir/summaries_zfix_pm/summary_pm_plus_*.csv")
    minus = load_masses("data_nir/summaries_zfix_pm/summary_pm_minus_*.csv")

    rec = []
    for _, r in sub.iterrows():
        i = r.ls_id
        if i in plus and i in minus and i in cen and r.sig_z > 0:
            lp, lm = plus[i][0], minus[i][0]
            rec.append({
                "ls_id": i, "z": r.z_best, "sig_z": r.sig_z,
                "kappa_max": r.kappa_max,
                "logm0": cen[i][0], "post_half": cen[i][1],
                "dm_dz": (lp - lm) / (2.0 * r.sig_z),
                "sig_m": 0.5 * abs(lp - lm)})
    d = pd.DataFrame(rec)
    print(f"{len(d)} galaxies with central and +/- fits "
          f"(of {len(sub)} subsample)")

    # analytic luminosity-distance expectation at each z
    z = d.z.to_numpy(float)
    dz = 0.01
    dl0 = COSMO.luminosity_distance(z).value
    dl1 = COSMO.luminosity_distance(z + dz).value
    d["dm_dz_analytic"] = 2.0 * (np.log10(dl1) - np.log10(dl0)) / dz

    exceeds = d.sig_m > d.post_half
    w = np.where(np.isfinite(d.kappa_max) & (d.kappa_max > 0),
                 d.kappa_max, 0.0)
    out = {
        "n": int(len(d)),
        "dm_dz": {"median": float(d.dm_dz.median()),
                  "nmad": nmad(d.dm_dz),
                  "median_analytic": float(d.dm_dz_analytic.median())},
        "sigma_m_photz": {
            "median": float(d.sig_m.median()),
            "p90": float(d.sig_m.quantile(0.9)),
            "kappa_weighted_mean":
                float(np.average(d.sig_m, weights=w)) if w.sum() else None,
            "frac_exceeds_posterior": float(exceeds.mean()),
            "median_posterior_half": float(d.post_half.median())},
        "z_bins": {},
        "response_curve": {},
    }
    for lo, hi in ((0.05, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.2)):
        s = d[(d.z >= lo) & (d.z < hi)]
        if len(s) < 15:
            continue
        out["z_bins"][f"{lo}-{hi}"] = {
            "n": int(len(s)),
            "dm_dz_median": float(s.dm_dz.median()),
            "dm_dz_analytic": float(s.dm_dz_analytic.median()),
            "sig_m_median": float(s.sig_m.median())}
        out["response_curve"][f"{lo}-{hi}"] = float(
            s.dm_dz.abs().median())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"dlogM/dz median {out['dm_dz']['median']:+.2f} "
          f"(analytic {out['dm_dz']['median_analytic']:+.2f})")
    print(f"sigma_M(photz) median {out['sigma_m_photz']['median']:.3f} "
          f"p90 {out['sigma_m_photz']['p90']:.3f} dex; exceeds FB "
          f"posterior for {100*out['sigma_m_photz']['frac_exceeds_posterior']:.0f}%")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
