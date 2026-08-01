#!/usr/bin/env python
"""Second-moment + skewness lensing channel (independent of the slope).

Lensing adds redshift-growing variance AND negative skewness (a bright
magnified tail) to the Hubble residuals. Neither statistic uses the
per-sight-line cross-correlation, so they are detection channels
independent of the slope fit.

Per broad z bin (equal-count):
- excess variance: Var_obs(hr) - [<MUERR^2> - <(0.055 z)^2>] -- the
  no-lensing expectation removes DES's sigma_lens term from MUERR so the
  lensing dispersion is not double-counted (same subtlety as the
  de-lensed-weights robustness row). Fit amplitudes against (i) the
  N-body law (0.055 z)^2 -> a_tot, and (ii) the catalog-visible
  prediction (2.171 A)^2 (tau_b^2 + <s2_berk>) -> a_cat, with
  tau_b^2 from snkappa.latent.bin_tau2 (classical noise subtracted).
- skewness: observed per-bin skew vs the catalog prediction
  skew_pred = (B A)^3 mu3(kappa)/sigma_obs^3, mu3 from the per-bin
  kappa_ext sample (classical prediction noise is ~symmetric, so the
  third moment of x tracks the third moment of the visible field).
  Fit a_skew; a_skew > 0 at significance = the skewness detection.

Run AFTER latent_fit.py --noise:  .venv/bin/python scripts/moments_test.py
Writes output/des_full/moments_test.json and output/figs/fig4_moments.pdf.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa.latent import SLOPE_TH, bin_tau2
from latent_fit import load_merged

A_FID = 0.79
N_BINS = 8
OUT = Path("output/des_full/moments_test.json")
FIGDIR = Path("output/figs"); FIGDIR.mkdir(parents=True, exist_ok=True)
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#888888"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "figure.dpi": 150,
})


def wls_amplitude(pred, obs, err, rng, n_boot=4000):
    """obs = a * pred, weighted by 1/err^2; bootstrap error over bins."""
    w = 1.0 / np.asarray(err) ** 2
    a = float(np.sum(w * pred * obs) / np.sum(w * pred ** 2))
    n = len(obs)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        boot[k] = (np.sum(w[i] * pred[i] * obs[i])
                   / max(np.sum(w[i] * pred[i] ** 2), 1e-30))
    return a, float(np.std(boot))


def wls_two_param(base, pred, obs, err, rng, n_boot=4000):
    """obs = c0*base + a*pred (weighted): frees the overall MUERR
    calibration (c0) so `a` measures only the z-GROWING lensing part.
    BBC's sigma_int is tuned so <MUERR^2> tracks the total variance by
    construction; without c0 the amplitude is degenerate with that
    calibration. Returns (a, a_err, c0)."""
    w = 1.0 / np.asarray(err) ** 2
    X = np.column_stack([np.asarray(base), np.asarray(pred)])

    def solve(Xs, ys, ws):
        beta, *_ = np.linalg.lstsq(Xs * np.sqrt(ws)[:, None],
                                   ys * np.sqrt(ws), rcond=None)
        return beta

    b = solve(X, np.asarray(obs), w)
    n = len(obs)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        boot[k] = solve(X[i], np.asarray(obs)[i], w[i])[1]
    return float(b[1]), float(np.std(boot)), float(b[0])


def quantile_skew(a):
    """(q84 - q50) - (q50 - q16), normalized: robust to far tails."""
    q16, q50, q84 = np.percentile(a, [16, 50, 84])
    return ((q84 - q50) - (q50 - q16)) / max(q84 - q16, 1e-12)


def main():
    d = load_merged()
    rng = np.random.default_rng(2718)
    B = SLOPE_TH  # -2.171

    # broad equal-count z bins for stable 2nd/3rd moments
    edges = np.quantile(d.zHD, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1e-6
    lab = np.digitize(d.zHD.to_numpy(), edges) - 1

    s2_cl = (d.rvar_gal_cl + d.rvar_cl_cl + d.var_zp).to_numpy()
    s2_bk = (np.clip(d.rvar_gal_tot - d.rvar_gal_cl, 0, None)
             + np.clip(d.rvar_cl_tot - d.rvar_cl_cl, 0, None)).to_numpy()
    tau2 = bin_tau2(d.kappa_ext.to_numpy(), s2_cl, lab)
    x = d.kappa_ext.to_numpy()
    y = d.hr.to_numpy()
    muerr2 = d.MUERR.to_numpy() ** 2
    zz = d.zHD.to_numpy()

    rows = []
    for b in range(N_BINS):
        m = lab == b
        n = int(m.sum())
        yb = y[m]
        # faint-clip: drop the faintest 1% as residual contamination
        # (BEAMS P>0.9 leaves ~1% CC, all in the FAINT tail -- opposite
        # in sign to the bright lensing tail)
        yc = yb[yb < np.quantile(yb, 0.99)]
        var_obs = float(yb.var(ddof=1))
        base = float(muerr2[m].mean() - np.mean((0.055 * zz[m]) ** 2))
        vb = np.empty(500); sb = np.empty(500); qb = np.empty(500)
        for k in range(500):
            i = rng.integers(0, n, n)
            ybi = yb[i]
            vb[k] = ybi.var(ddof=1)
            yci = ybi[ybi < np.quantile(ybi, 0.99)]
            sb[k] = sps.skew(yci, bias=False)
            qb[k] = quantile_skew(ybi)
        pred_var = (B * A_FID) ** 2 * (tau2[m].mean() + s2_bk[m].mean())
        mu3 = float(np.mean((x[m] - x[m].mean()) ** 3))
        skew_pred = (B * A_FID) ** 3 * mu3 / max(var_obs, 1e-12) ** 1.5
        # predicted quantile skew: MC of B*A*kappa + Gaussian noise
        ysim = (B * A_FID * rng.choice(x[m], 4000)
                + rng.standard_normal(4000) * np.sqrt(var_obs - pred_var))
        rows.append({
            "zbar": float(zz[m].mean()), "n": n,
            "var_obs": var_obs, "var_err": float(vb.std()),
            "base_muerr": base,
            "pred_var_nbody": float(np.mean((0.055 * zz[m]) ** 2)),
            "pred_var_catalog": float(pred_var),
            "skew_obs_clip": float(sps.skew(yc, bias=False)),
            "skew_err": float(sb.std()),
            "skew_pred_catalog": float(skew_pred),
            "qskew_obs": quantile_skew(yb), "qskew_err": float(qb.std()),
            "qskew_pred": quantile_skew(ysim)})

    t = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    # variance: Var_obs = c0 * <MUERR^2 - (0.055z)^2> + a * law(z)
    a_tot, e_tot, c0_tot = wls_two_param(
        t["base_muerr"], t["pred_var_nbody"], t["var_obs"],
        t["var_err"], rng)
    a_cat, e_cat, c0_cat = wls_two_param(
        t["base_muerr"], t["pred_var_catalog"], t["var_obs"],
        t["var_err"], rng)
    a_skw, e_skw = wls_amplitude(t["skew_pred_catalog"],
                                 t["skew_obs_clip"], t["skew_err"], rng)
    a_qsk, e_qsk = wls_amplitude(t["qskew_pred"], t["qskew_obs"],
                                 t["qskew_err"], rng)

    out = {"n_sn": len(d), "A_fid": A_FID, "n_bins": N_BINS, "bins": rows,
           "a_var_vs_nbody_0.055z": [a_tot, e_tot, c0_tot],
           "a_var_vs_catalog": [a_cat, e_cat, c0_cat],
           "a_skew_faintclip_vs_catalog": [a_skw, e_skw],
           "a_quantile_skew_vs_catalog": [a_qsk, e_qsk],
           "note": ("variance model frees the MUERR calibration (c0): "
                    "BBC's sigma_int makes <MUERR^2> track total "
                    "variance by construction, so only the z-growing "
                    "part is identifiable. Moment skew is faint-1%-"
                    "clipped (CC contamination is faint-tailed, "
                    "opposite to lensing); quantile skew is tail-"
                    "robust. hr per-fine-bin demeaned upstream.")}
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print(f"variance: a_tot(0.055z law) = {a_tot:.2f} +- {e_tot:.2f} "
          f"(c0={c0_tot:.2f}) | a_cat = {a_cat:.2f} +- {e_cat:.2f} "
          f"(c0={c0_cat:.2f})")
    print(f"skewness: moment(faint-clip) a = {a_skw:.2f} +- {e_skw:.2f} "
          f"({a_skw/max(e_skw,1e-9):.1f} sig) | quantile a = "
          f"{a_qsk:.2f} +- {e_qsk:.2f} ({a_qsk/max(e_qsk,1e-9):.1f} sig)")

    # ---------------------------------------------------------- figure --
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    excess_cal = t["var_obs"] - c0_tot * t["base_muerr"]
    ax.errorbar(t["zbar"], 1e3 * excess_cal, yerr=1e3 * t["var_err"],
                fmt="o", color="k", ms=4, capsize=2,
                label="calibrated excess variance")
    zg = np.linspace(0.1, 1.15, 50)
    ax.plot(zg, 1e3 * a_tot * (0.055 * zg) ** 2, "-", color=BLUE, lw=1.5,
            label=rf"$a\,(0.055z)^2$, $a={a_tot:.2f}\pm{e_tot:.2f}$")
    ax.plot(t["zbar"], 1e3 * t["pred_var_catalog"], "s--", color=VERM,
            ms=3.5, lw=1.2, label="catalog-visible prediction")
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"excess Var($\Delta\mu$) [$10^{-3}$ mag$^2$]")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.errorbar(t["zbar"], t["skew_obs_clip"], yerr=t["skew_err"],
                fmt="o", color="k", ms=4, capsize=2,
                label="observed skew (faint 1% clipped)")
    ax.plot(t["zbar"], a_skw * t["skew_pred_catalog"], "s--", color=VERM,
            ms=3.5, lw=1.2,
            label=rf"$a\times$pred, $a={a_skw:.1f}\pm{e_skw:.1f}$")
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlabel(r"$z$"); ax.set_ylabel(r"skew($\Delta\mu$)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_moments.pdf")
    print(f"saved {OUT} and {FIGDIR/'fig4_moments.pdf'}")


if __name__ == "__main__":
    main()
