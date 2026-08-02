#!/usr/bin/env python
"""Union3 Hubble diagram from the released UNITY inputs (build Phase 1).

UNITY does not release per-SN standardized distances (its distance
product is the 22-node binned spline), so this script Tripp-standardizes
the released light-curve fits:

    mu_i = mB_i + alpha x1_i - beta c_i - M - gamma step(mass > 10)

with (alpha, beta, gamma, M) fit by weighted least squares against flat
LCDM (Om = 0.356, Union3's own fit -- the choice is absorbed by the
per-z-bin demeaning applied downstream) and a PER-SAMPLE sigma_int
(iterative variance matching; the 25 source samples are heterogeneous).
Per-SN measurement variance comes from the full mBx1c covariance.
MUERR additionally carries sigma_lens = 0.055 z (matching the DES-SN5YR
convention that downstream scripts assume) and a 300 km/s peculiar-
velocity term. 3.5-sigma outliers are clipped (counted) and the fit
repeated once.

Output: output/union3/union3_hd.csv with DES-compatible columns
(CID, sample, HOST_RA/HOST_DEC, zHD, MU, MUERR, HOST_LOGMASS,
PROBIA = 1.0, cluster_targeted flag for See-Change).

Run: .venv/bin/python scripts/union3_prep.py
"""

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.regions import parse_mixed_coords

PICKLE = Path("data_union3/inputs_union3.pickle")
OUT = Path("output/union3/union3_hd.csv")
OM_FID = 0.356
SIG_LENS = 0.055
SIG_PV_KMS = 300.0
C_KMS = 299792.458
MASS_STEP_AT = 10.0
CLIP_SIGMA = 3.5


def load_inputs():
    d0 = pickle.load(gzip.open(PICKLE, "rb"))[0]
    ra, dec = parse_mixed_coords(d0["RA"], d0["Dec"])
    return pd.DataFrame({
        "CID": [p.rstrip("/").split("/")[-1] for p in d0["snpaths"]],
        "sample": [p.rstrip("/").split("/")[-2] for p in d0["snpaths"]],
        "HOST_RA": ra, "HOST_DEC": dec,
        "zHD": d0["z_CMB_list"], "z_helio": d0["z_helio_list"],
        "mB": d0["mB_list"], "x1": d0["x1_list"], "c": d0["c_list"],
        "HOST_LOGMASS": [float(m) for m in d0["mass"]],
    }), d0["mBx1c_cov_list"]


def tripp_fit(df, cov, mask):
    """Iterative WLS Tripp fit with per-sample sigma_int.

    Returns (params dict, mu, muerr, resid) over ALL rows (mask selects
    the rows used in the fit)."""
    cosmo = FlatLambdaCDM(H0=70.0, Om0=OM_FID)
    mu_lcdm = cosmo.distmod(df.zHD.to_numpy()).value
    step = (df.HOST_LOGMASS.to_numpy() > MASS_STEP_AT).astype(float)
    x1 = df.x1.to_numpy(); cc = df.c.to_numpy(); mB = df.mB.to_numpy()
    samples = df["sample"].to_numpy()
    sig_pv = (5.0 / np.log(10.0)) * SIG_PV_KMS / (
        C_KMS * df.zHD.to_numpy())

    sig_int = {s: 0.12 for s in np.unique(samples)}
    alpha, beta = 0.14, 3.1
    for _ in range(8):
        sig_meas2 = (cov[:, 0, 0] + alpha**2 * cov[:, 1, 1]
                     + beta**2 * cov[:, 2, 2] + 2 * alpha * cov[:, 0, 1]
                     - 2 * beta * cov[:, 0, 2]
                     - 2 * alpha * beta * cov[:, 1, 2])
        s_int = np.array([sig_int[s] for s in samples])
        var = np.clip(sig_meas2, 1e-4, None) + s_int**2 + sig_pv**2
        w = np.where(mask, 1.0 / var, 0.0)
        # linear WLS: mB - mu_lcdm = -alpha x1 + beta c + M + gamma step
        X = np.column_stack([-x1, cc, np.ones_like(x1), step])
        y = mB - mu_lcdm
        A = X.T @ (X * w[:, None]); b = X.T @ (w * y)
        alpha, beta, M, gamma = np.linalg.solve(A, b)
        resid = y - X @ np.array([alpha, beta, M, gamma])
        # per-sample sigma_int: match unit reduced chi2
        for s in sig_int:
            m = mask & (samples == s)
            if m.sum() < 5:
                sig_int[s] = 0.12
                continue
            excess = np.mean(resid[m]**2 - sig_meas2[m] - sig_pv[m]**2)
            sig_int[s] = float(np.sqrt(np.clip(excess, 0.01**2, 0.35**2)))
    mu = mB + alpha * x1 - beta * cc - M - gamma * step
    muerr = np.sqrt(var + (SIG_LENS * df.zHD.to_numpy())**2)
    params = {"alpha": float(alpha), "beta": float(beta), "M": float(M),
              "gamma_mass_step": float(gamma),
              "sig_int_by_sample": {k: round(v, 3)
                                    for k, v in sig_int.items()}}
    return params, mu, muerr, resid / np.sqrt(var)


def main():
    df, cov = load_inputs()
    print(f"{len(df)} SNe from {df['sample'].nunique()} samples")
    mask = np.ones(len(df), dtype=bool)
    params, mu, muerr, pull = tripp_fit(df, cov, mask)
    clip = np.abs(pull) > CLIP_SIGMA
    print(f"pass 1: alpha={params['alpha']:.3f} beta={params['beta']:.2f} "
          f"gamma={params['gamma_mass_step']:+.3f}; "
          f"clipping {int(clip.sum())} outliers (> {CLIP_SIGMA} sigma)")
    params, mu, muerr, pull = tripp_fit(df, cov, mask & ~clip)
    print(f"pass 2: alpha={params['alpha']:.3f} beta={params['beta']:.2f} "
          f"gamma={params['gamma_mass_step']:+.3f}")

    out = df[["CID", "sample", "HOST_RA", "HOST_DEC", "zHD",
              "HOST_LOGMASS"]].copy()
    out["MU"] = mu
    out["MUERR"] = muerr
    out["PROBIA"] = 1.0
    out["clipped"] = clip
    out["cluster_targeted"] = df["sample"] == "SuzukiRubin"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    import json
    Path("output/union3/tripp_params.json").write_text(
        json.dumps(params, indent=2))
    hr = mu - FlatLambdaCDM(H0=70, Om0=OM_FID).distmod(
        out.zHD.to_numpy()).value
    keep = ~clip & (out.zHD > 0.1)
    print(f"usable (z>0.1, unclipped): {int(keep.sum())}; "
          f"HR rms = {hr[keep].std():.3f} mag; "
          f"median MUERR = {np.median(muerr[keep]):.3f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
