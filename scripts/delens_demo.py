#!/usr/bin/env python
"""De-lensing demonstration: subtract the measured lensing signal and
quantify what changes (TODO/plan Item D).

- De-lens with the measured amplitude: MU' = MU + 2.171 * A * kappa_ext
  (linear) and MU'' = MU - A * dmu_pred (exact kappa+shear prediction,
  when the dmu_pred column exists).
- Differential cosmology: flat LCDM (Om) and flat wCDM (Om, w) fits with
  diagonal MUERR and the absolute offset analytically profiled. NO
  covariance matrix -- absolute values are not the DES-SN5YR fit; only
  the DIFFERENTIAL shifts between original and de-lensed distances are
  the deliverable.
- Tail control: HR skewness at z > 0.7 and the count of >2.5 sigma
  BRIGHT outliers, before vs after; per-sightline shifts for the top-10
  kappa_ext objects (the manuscript's Table 1).
- Verification: A = 0 reproduces the baseline exactly.

Run: .venv/bin/python scripts/delens_demo.py
Writes output/des_full/delens_demo.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, FlatwCDM
from scipy import stats as sps
from scipy.optimize import minimize_scalar, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

A_HAT = 0.79
SLOPE_TH = -5.0 / np.log(10.0)
OUT = Path("output/des_full/delens_demo.json")


def chi2(mu, muerr, zz, cosmo):
    """Diagonal chi^2 with the absolute offset profiled analytically."""
    r = mu - cosmo.distmod(zz).value
    w = 1.0 / muerr ** 2
    off = np.sum(w * r) / np.sum(w)
    return float(np.sum(w * (r - off) ** 2))


def fit_lcdm(mu, muerr, zz):
    f = minimize_scalar(
        lambda om: chi2(mu, muerr, zz, FlatLambdaCDM(H0=70, Om0=om)),
        bounds=(0.1, 0.6), method="bounded",
        options={"xatol": 1e-5})
    return float(f.x)


def fit_wcdm(mu, muerr, zz):
    f = minimize(
        lambda p: chi2(mu, muerr, zz,
                       FlatwCDM(H0=70, Om0=p[0], w0=p[1])),
        x0=[0.35, -1.0], method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6})
    return float(f.x[0]), float(f.x[1])


def tail_stats(hr, muerr, zz):
    m = zz > 0.7
    r = hr[m] / muerr[m]
    return {
        "n_z_gt_0.7": int(m.sum()),
        "skew_z_gt_0.7": float(sps.skew(hr[m], bias=False)),
        "n_bright_2.5sig": int(np.sum(r < -2.5)),
        "n_faint_2.5sig": int(np.sum(r > 2.5)),
    }


def main():
    d = pd.read_csv("output/des_full/des_all_kappa.csv")
    g = d[d.PROBIA > 0.9].reset_index(drop=True)
    mu = g.MU.to_numpy()
    muerr = g.MUERR.to_numpy()
    zz = g.zHD.to_numpy()
    kext = g.kappa_ext.to_numpy()
    hr = g.hr.to_numpy()

    variants = {"baseline": np.zeros_like(kext),
                "linear": -SLOPE_TH * A_HAT * kext}
    if "dmu_pred" in g:
        variants["exact"] = -A_HAT * g.dmu_pred.to_numpy()
    # verification: A=0 must reproduce baseline exactly
    variants["A0_check"] = -SLOPE_TH * 0.0 * kext

    out = {"A_hat": A_HAT, "n_sn": len(g), "fits": {}}
    for name, corr in variants.items():
        mu_c = mu + corr
        om = fit_lcdm(mu_c, muerr, zz)
        om_w, w0 = fit_wcdm(mu_c, muerr, zz)
        out["fits"][name] = {
            "Om_lcdm": om, "Om_wcdm": om_w, "w": w0,
            "tail": tail_stats(hr + corr, muerr, zz),
            "sigma_correction_mag": float(np.std(corr)),
        }
        print(f"{name:9s} Om(LCDM)={om:.4f}  (Om,w)=({om_w:.4f},{w0:.4f}) "
              f"| z>0.7 skew {out['fits'][name]['tail']['skew_z_gt_0.7']:+.3f}"
              f" bright>2.5s {out['fits'][name]['tail']['n_bright_2.5sig']}")

    base = out["fits"]["baseline"]
    assert out["fits"]["A0_check"]["Om_lcdm"] == base["Om_lcdm"]
    for name in ("linear", "exact"):
        if name not in out["fits"]:
            continue
        f = out["fits"][name]
        out[f"delta_{name}"] = {
            "dOm_lcdm": f["Om_lcdm"] - base["Om_lcdm"],
            "dOm_wcdm": f["Om_wcdm"] - base["Om_wcdm"],
            "dw": f["w"] - base["w"],
            "dskew_z_gt_0.7": (f["tail"]["skew_z_gt_0.7"]
                               - base["tail"]["skew_z_gt_0.7"]),
            "dn_bright": (f["tail"]["n_bright_2.5sig"]
                          - base["tail"]["n_bright_2.5sig"]),
        }
        print(f"delta_{name}: dOm={out[f'delta_{name}']['dOm_lcdm']:+.5f} "
              f"dw={out[f'delta_{name}']['dw']:+.5f} "
              f"dskew={out[f'delta_{name}']['dskew_z_gt_0.7']:+.3f} "
              f"dN_bright={out[f'delta_{name}']['dn_bright']:+d}")

    # top-10 kappa sightlines: per-object corrections
    top = g.nlargest(10, "kappa_ext")
    out["top10"] = [{
        "CID": int(r.CID), "zHD": float(r.zHD),
        "kappa_ext": float(r.kappa_ext), "hr": float(r.hr),
        "corr_linear_mag": float(-SLOPE_TH * A_HAT * r.kappa_ext),
        "corr_exact_mag": (float(-A_HAT * r.dmu_pred)
                           if "dmu_pred" in g else None),
    } for r in top.itertuples()]

    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
