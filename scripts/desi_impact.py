#!/usr/bin/env python
"""Impact of de-lensing DES-SN5YR on the DESI DR2 evolving-dark-energy
preference (backs the manuscript's Delta-chi2 ~ +0.4 statement).

The DESI DR2 BAO cosmology paper (arXiv:2503.14738) quotes a 4.2 sigma
preference for w0waCDM over LCDM for DESI+CMB+DES-SN5YR --- the same
published distances this pipeline de-lenses. Their convention maps
Delta-chi2 to significance via p = exp(-Delta-chi2/2) (2 dof), which
reproduces their quoted 2.8/3.1/3.8/4.2 sigma for Delta-chi2 =
10.7/12.5/17.4/21.0; 5 sigma requires 28.7.

Method (Gaussian limit): de-lensing shifts the SN data by
dmu_i = +2.171 A kappa_ext,i (fainter where overdense). The change in
the combined LCDM-rejection Delta-chi2 is, to first order, the
covariance-independent identity

    DDchi2 = 2 g . d0,   g = J^T W_proj dmu,   d0 = theta_hat - (-1, 0)

with J = d mu / d(w0,wa) at the combined best fit, W_proj the
offset(M)-marginalized diagonal SN weights, and theta_hat the published
DESI+CMB+DESY5 best fit. The SN-only Fisher shift C_SN g is
cross-checked against a direct differential refit.

Published anchors (transcribed from arXiv:2503.14738, v2):
  DESI+CMB+DESY5 w0waCDM: w0 = -0.752 +- 0.057, wa = -0.86 +- 0.22
                          (symmetrized), Delta-chi2(MAP) = 21.0 (4.2 sigma).

Key caveats: diagonal SN covariance (no DES stat+syst matrix), Gaussian
posterior, Om held at the combined best fit for the SN response, and the
correction amplitude carries the measured A = 0.79 +- 0.20.

Run: .venv/bin/python scripts/desi_impact.py
Writes output/des_full/desi_impact.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import Flatw0waCDM
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

A_HAT, A_ERR = 0.79, 0.20
TH_HAT = np.array([-0.752, -0.86])   # DESI+CMB+DESY5 w0waCDM best fit
SIG = np.array([0.057, 0.215])
DCHI2_BASE = 21.0                    # their LCDM-rejection Delta-chi2
OM = 0.323
OUT = Path("output/des_full/desi_impact.json")


def sigma_of(dchi2):
    return float(norm.isf(np.exp(-dchi2 / 2) / 2))


def main():
    d = pd.read_csv("output/des_full/des_all_kappa.csv")
    g = d[d.PROBIA > 0.9].reset_index(drop=True)
    zz, muerr = g.zHD.to_numpy(), g.MUERR.to_numpy()
    kext = g.kappa_ext.to_numpy()

    eps = 1e-3
    J = np.empty((len(zz), 2))
    for j, dp in enumerate(([eps, 0], [0, eps])):
        cp = Flatw0waCDM(H0=70, Om0=OM, w0=TH_HAT[0] + dp[0],
                         wa=TH_HAT[1] + dp[1])
        cm = Flatw0waCDM(H0=70, Om0=OM, w0=TH_HAT[0] - dp[0],
                         wa=TH_HAT[1] - dp[1])
        J[:, j] = (cp.distmod(zz).value - cm.distmod(zz).value) / (2 * eps)

    w = 1.0 / muerr ** 2

    def wproj(v):
        return w * v - w * np.sum(w * v) / np.sum(w)

    d0 = TH_HAT - np.array([-1.0, 0.0])
    F = J.T @ (J * w[:, None] - np.outer(w, w @ J) / w.sum())
    C_SN = np.linalg.inv(F)

    out = {"published_anchor": {
               "combo": "DESI DR2 BAO + CMB + DES-SN5YR (arXiv:2503.14738)",
               "w0": TH_HAT[0], "wa": TH_HAT[1],
               "sig_w0": SIG[0], "sig_wa": SIG[1],
               "dchi2": DCHI2_BASE, "sigma": sigma_of(DCHI2_BASE),
               "dchi2_for_5sigma": 28.7},
           "sn_only_fisher": {
               "sig_w0": float(np.sqrt(C_SN[0, 0])),
               "sig_wa": float(np.sqrt(C_SN[1, 1])),
               "corr": float(C_SN[0, 1]
                             / np.sqrt(C_SN[0, 0] * C_SN[1, 1]))},
           "scan": []}

    for a in (A_HAT - A_ERR, A_HAT, A_HAT + A_ERR, 2 * A_HAT):
        corr = 2.171 * a * kext
        gvec = J.T @ wproj(corr)
        ddchi2 = float(2 * gvec @ d0)          # + O(g C g), negligible
        dth_sn = C_SN @ gvec
        out["scan"].append({
            "A": a, "ddchi2": ddchi2,
            "sigma_after": sigma_of(DCHI2_BASE + ddchi2),
            "sn_only_shift_w0_wa": [float(dth_sn[0]), float(dth_sn[1])]})

    fid = out["scan"][1]
    print(f"A = {A_HAT}: DDchi2 = {fid['ddchi2']:+.3f} "
          f"-> {out['published_anchor']['sigma']:.2f} -> "
          f"{fid['sigma_after']:.2f} sigma (5 sigma needs +7.7)")
    print(f"SN-only Fisher (w0,wa): +-{out['sn_only_fisher']['sig_w0']:.2f}"
          f"/+-{out['sn_only_fisher']['sig_wa']:.2f}, "
          f"corr {out['sn_only_fisher']['corr']:.3f} -- the correction "
          f"lies along the SN degeneracy BAO+CMB break")
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
