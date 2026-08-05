#!/usr/bin/env python
"""Velocity-dispersion check of the halo rung (FSF P3).

Every existing validation tests the stellar-mass rung (M*) or the
integrated lensing signal (Delta-Sigma closure). This is the first
PER-GALAXY check of the halo rung itself: for spec-z kappa contributors
with a measured FastSpecFit velocity dispersion, compare the
SMHM-inverted halo mass <M_h|M*> against sigma_v, which never touches
the stellar mass.

Three statements, in increasing strength:
  1. slope: d log M_h / d log sigma vs the virial expectation (~3);
  2. scatter: NMAD of log M_h at fixed sigma (the halo-rung noise);
  3. absolute: M_h vs the singular-isothermal-sphere anchor
     M200_SIS = (2 sqrt(2)/10) sigma^3 / (G H(z))  — an analytic
     normalization good to ~0.1-0.2 dex, quoted as such.

Quality cuts: vdisp in (100, 420) km/s excluding the modal pinned
values (FSF fixes vdisp at its prior default for low-S/N fits — these
show up as a delta-function spike and carry no information), z_spec
0.05-0.6, and membership in the kappa-contributor list.

Inputs: data_nir/fsf_masses_override.csv, data_nir/fb_targets_all.csv
Output: output/nir_video/vdisp_halos.json (tracked)

Run: .venv/bin/python scripts/fsf_vdisp.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.halos import HaloModel
from snkappa.config import HaloModelConfig

OUT = Path("output/nir_video/vdisp_halos.json")
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.352)


def nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size \
        else float("nan")


def m200_sis(sigma_kms, z):
    """SIS M200 [Msun]: M = 2 sigma^2 r200 / G, r200 = sqrt(2) sigma
    / (10 H(z))."""
    sig = np.asarray(sigma_kms, float) * u.km / u.s
    hz = COSMO.H(np.asarray(z, float))
    m = (2.0 * np.sqrt(2.0) / 10.0) * sig ** 3 / (const.G * hz)
    return m.to(u.Msun).value


def main():
    fsf = pd.read_csv("data_nir/fsf_masses_override.csv",
                      dtype={"ls_id": str})
    tg = pd.read_csv("data_nir/fb_targets_all.csv", dtype={"ls_id": str})
    d = fsf.merge(tg[["ls_id", "kappa_max", "kappa_sum"]], on="ls_id")
    print(f"{len(d)} spec contributors with FSF masses")

    v = d.vdisp.to_numpy(float)
    # pinned-prior spikes: any single value carrying >2% of the sample
    vals, counts = np.unique(np.round(v[np.isfinite(v)], 1),
                             return_counts=True)
    pinned = set(vals[counts > 0.02 * np.isfinite(v).sum()])
    ok = (np.isfinite(v) & (v > 100) & (v < 420)
          & ~np.isin(np.round(v, 1), list(pinned))
          & (d.z_spec > 0.05) & (d.z_spec < 0.6))
    s = d[ok].copy()
    print(f"{len(s)} pass vdisp quality cuts "
          f"(pinned spikes excluded: {sorted(pinned)[:4]}...)")
    if len(s) < 50:
        raise SystemExit("too few galaxies for the vdisp check")

    hm = HaloModel(HaloModelConfig(), COSMO, 1.15)
    ib = hm.zbin_index(s.z_spec.to_numpy(float))
    # the chain's Eddington-debiased posterior inversion, per z bin
    logmh = np.array([float(hm._inv[int(i)](m))
                      for i, m in zip(ib, s.logm_p50.to_numpy(float))])

    logsig = np.log10(s.vdisp.to_numpy(float))
    # 1. slope
    A = np.vstack([logsig, np.ones_like(logsig)]).T
    coef, *_ = np.linalg.lstsq(A, logmh, rcond=None)
    slope = float(coef[0])
    # robust slope via binned medians
    qs = np.quantile(logsig, np.linspace(0, 1, 7))
    bin_med = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (logsig >= lo) & (logsig < hi)
        if m.sum() > 20:
            bin_med.append((float(np.median(logsig[m])),
                            float(np.median(logmh[m])), int(m.sum())))
    if len(bin_med) >= 3:
        bx = np.array([b[0] for b in bin_med])
        by = np.array([b[1] for b in bin_med])
        slope_binned = float(np.polyfit(bx, by, 1)[0])
    else:
        slope_binned = float("nan")

    # 2. scatter at fixed sigma
    resid = logmh - (coef[0] * logsig + coef[1])
    # 3. absolute vs SIS
    logm_sis = np.log10(m200_sis(s.vdisp, s.z_spec))
    dsis = logmh - logm_sis

    w = s.kappa_sum.to_numpy(float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    out = {
        "n_used": int(len(s)),
        "vdisp_range": [float(s.vdisp.min()), float(s.vdisp.max())],
        "slope_logmh_logsigma": {
            "lstsq": slope, "binned_medians": slope_binned,
            "virial_expectation": 3.0,
            "bins": bin_med},
        "scatter_logmh_at_fixed_sigma_nmad": nmad(resid),
        "vs_SIS": {
            "median_logmh_minus_logm_sis": float(np.median(dsis)),
            "nmad": nmad(dsis),
            "kappa_weighted_mean": float(np.average(dsis, weights=w))
            if w.sum() else None,
            "note": ("SIS normalization is analytic (~0.1-0.2 dex); "
                     "DESI fiber sigma is central, not sigma_e "
                     "(aperture correction ~5-10% not applied)")},
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"slope d(logMh)/d(logsig): lstsq {slope:.2f}, "
          f"binned {slope_binned:.2f} (virial ~3)")
    print(f"scatter logMh|sigma: {nmad(resid):.3f} dex")
    print(f"vs SIS: median {np.median(dsis):+.3f} dex, NMAD {nmad(dsis):.3f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
