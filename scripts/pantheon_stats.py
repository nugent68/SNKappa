#!/usr/bin/env python
"""Pantheon+ capstone statistics (Phase 4).

1. GLS slope with the full released stat+syst covariance -- the fit the
   Pantheon+ README requires -- vs the diagonal fit (quantifies the
   'diagonal errors' caveat carried by all previous slopes).
2. Standardization test at fixed kappa: Pantheon+ vs Union3 residuals on
   position-matched sightlines; the difference regression
   d(hr) = (hr_P+ - hr_U3) vs kappa isolates differential
   standardization systematics (zero if the lensing signal is
   standardization-independent).
3. DESI DR2 trio closer: DDchi2 for the Pantheon+ leg (2.8 sigma;
   anchors from arXiv:2503.14738: w0 = -0.838 +- 0.055,
   wa = -0.62 +- 0.21, Delta-chi2 = 10.7).
4. Cross-compilation kappa correlation on matched sightlines.

Run after union3_full.py --hd pantheon:
    .venv/bin/python scripts/pantheon_stats.py
Writes output/pantheon/gls_fit.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.fitting import bootstrap_slope, gls_slope
from snkappa.kappa import angular_sep_arcsec

COV = Path("data_pantheon/Pantheon+SH0ES_STAT+SYS.cov")
OUT = Path("output/pantheon/gls_fit.json")
OM_FID = 0.334          # Pantheon+ flat-LCDM fit; absorbed by intercept
SLOPE_TH = -5.0 / np.log(10.0)
# DESI DR2 + CMB + Pantheon+ anchors (arXiv:2503.14738)
TH_HAT = np.array([-0.838, -0.62])
DCHI2_BASE = 10.7


def load_cov(rows):
    raw = np.loadtxt(COV)
    n = int(raw[0])
    C = raw[1:].reshape(n, n)
    return C[np.ix_(rows, rows)]


def main():
    p = pd.read_csv("output/pantheon/pantheon_kappa.csv")
    good = p[~p.clipped & ~p.cluster_targeted].reset_index(drop=True)
    rng = np.random.default_rng(6180)
    out = {"n_good": len(good)}

    # ---- 1. GLS with full covariance vs diagonal ------------------------
    x = good.kappa_ext.to_numpy()
    cosmo = FlatLambdaCDM(H0=70, Om0=OM_FID)
    y_raw = good.MU.to_numpy() - cosmo.distmod(good.zHD.to_numpy()).value
    rows = good.row_index.to_numpy().astype(int)
    C = load_cov(rows)
    b_gls, e_gls = gls_slope(x, y_raw, C)
    b_dgl, e_dgl = gls_slope(x, y_raw, np.diag(np.diag(C)))
    b_std, e_std = bootstrap_slope(x, good.hr.to_numpy(),
                                   good.MUERR.to_numpy(), rng)
    out["slope_gls_fullcov"] = [b_gls, e_gls]
    out["slope_gls_diag_of_cov"] = [b_dgl, e_dgl]
    out["slope_diag_binned_hr"] = [b_std, e_std]
    out["A_gls"] = [b_gls / SLOPE_TH, e_gls / abs(SLOPE_TH)]
    print(f"GLS full stat+syst cov: slope {b_gls:+.2f} +- {e_gls:.2f} "
          f"({abs(b_gls)/e_gls:.1f} sig)  A = {b_gls/SLOPE_TH:.2f} "
          f"+- {e_gls/abs(SLOPE_TH):.2f}")
    print(f"  diagonal-of-cov:      slope {b_dgl:+.2f} +- {e_dgl:.2f}")
    print(f"  usual binned-hr fit:  slope {b_std:+.2f} +- {e_std:.2f}")

    # ---- 2. standardization test at fixed kappa vs Union3 ---------------
    u3 = pd.read_csv("output/union3/union3_kappa.csv")
    u3 = u3[~u3.clipped & ~u3.cluster_targeted]
    mi, mj = [], []
    for i, r in good.iterrows():
        sep = angular_sep_arcsec(r.HOST_RA, r.HOST_DEC,
                                 u3.HOST_RA.to_numpy(),
                                 u3.HOST_DEC.to_numpy())
        j = int(np.argmin(sep))
        if sep[j] < 1.5 and abs(u3.zHD.iloc[j] - r.zHD) < 0.01:
            mi.append(i); mj.append(j)
    gp = good.loc[mi].reset_index(drop=True)
    gu = u3.iloc[mj].reset_index(drop=True)
    out["n_matched_union3"] = len(gp)
    kk = 0.5 * (gp.kappa_ext.to_numpy() + gu.kappa_ext.to_numpy())
    out["kappa_cross_corr"] = float(np.corrcoef(
        gp.kappa_ext, gu.kappa_ext)[0, 1])
    bp, ep = bootstrap_slope(kk, gp.hr.to_numpy(), gp.MUERR.to_numpy(),
                             rng)
    bu, eu = bootstrap_slope(kk, gu.hr.to_numpy(), gu.MUERR.to_numpy(),
                             rng)
    dhr = gp.hr.to_numpy() - gu.hr.to_numpy()
    sd = np.hypot(gp.MUERR, gu.MUERR).to_numpy()  # upper bound (shared LC)
    bd, ed = bootstrap_slope(kk, dhr, sd, rng)
    out["standardization_test"] = {
        "slope_pantheon_matched": [bp, ep],
        "slope_union3_matched": [bu, eu],
        "slope_hr_difference": [bd, ed]}
    print(f"matched sightlines: {len(gp)} | kappa corr "
          f"{out['kappa_cross_corr']:.3f}")
    print(f"  slope P+ {bp:+.2f}+-{ep:.2f} | U3 {bu:+.2f}+-{eu:.2f} | "
          f"difference {bd:+.2f}+-{ed:.2f} (0 = standardization-"
          f"independent)")

    # ---- 3. DESI trio closer --------------------------------------------
    zz = good.zHD.to_numpy()
    muerr = good.MUERR.to_numpy()
    corr = 2.171 * 0.78 * x            # joint amplitude
    eps = 1e-3
    J = np.empty((len(zz), 2))
    for j, dp in enumerate(([eps, 0], [0, eps])):
        cp = Flatw0waCDM(H0=70, Om0=0.31, w0=TH_HAT[0] + dp[0],
                         wa=TH_HAT[1] + dp[1])
        cm = Flatw0waCDM(H0=70, Om0=0.31, w0=TH_HAT[0] - dp[0],
                         wa=TH_HAT[1] - dp[1])
        J[:, j] = (cp.distmod(zz).value - cm.distmod(zz).value) / (2 * eps)
    w = 1.0 / muerr**2
    g = J.T @ (w * corr - w * np.sum(w * corr) / w.sum())
    d0 = TH_HAT - np.array([-1.0, 0.0])
    ddchi2 = float(2 * g @ d0)

    def sig_of(dc):
        return float(norm.isf(np.exp(-dc / 2) / 2))

    out["desi_pantheon_leg"] = {
        "ddchi2": ddchi2, "base_dchi2": DCHI2_BASE,
        "sigma_before": sig_of(DCHI2_BASE),
        "sigma_after": sig_of(DCHI2_BASE + ddchi2)}
    print(f"DESI Pantheon+ leg: DDchi2 = {ddchi2:+.3f} "
          f"({sig_of(DCHI2_BASE):.2f} -> "
          f"{sig_of(DCHI2_BASE + ddchi2):.2f} sigma)")

    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
