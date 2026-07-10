#!/usr/bin/env python
"""Paper statistics + figures for the ApJL manuscript.

Reads output/des_full/des_all_kappa.csv (+ variant CSVs if present); writes
fig1-3.pdf, stats.json, table1_top.csv into this directory. Palette:
Okabe-Ito blue/vermilion (CVD-validated); identity always carried by
linestyle + direct label too.

Statistics beyond the headline fit (TODO 1.1, 1.2b/c, 3.2, 3.6, 3.7):
- permutation null: Hubble residuals shuffled among SNe WITHIN z bins
  (preserves both marginals and the kappa spatial structure); empirical
  p-value of the slope. This absorbs the spatial covariance of overlapping
  apertures that an i.i.d. bootstrap ignores.
- spatial block bootstrap (0.5 deg blocks >= aperture scale) as cross-check.
- host-environment confounder tests: kappa_ext vs host stellar mass, and the
  slope refit with a host-mass-step covariate.
- robustness rows: z < 1.0, de-lensed weights (sigma_lens removed from
  MUERR in quadrature), and any variant runs found on disk.
- Table 1 gets the exact tail magnification -2.5 log10[(1-kappa)^-2]
  instead of the linearized -2.171 kappa.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

import sys
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from snkappa.fitting import bootstrap_slope  # noqa: E402

CSV = ROOT / "output/des_full/des_all_kappa.csv"
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#888888"
SLOPE_TH = -5.0 / np.log(10.0)   # dHR/dkappa = -2.171 (mu = 1+2k weak limit)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "figure.dpi": 150,
})

r = pd.read_csv(CSV)
rng = np.random.default_rng(1234)
good = r[r.PROBIA > 0.9].reset_index(drop=True)


def wslope(d, n_boot=4000):
    return bootstrap_slope(d.kappa_ext.to_numpy(), d.hr.to_numpy(),
                           d.MUERR.to_numpy(), rng, n_boot=n_boot)


def slope_only(x, y, sig):
    return float(np.polyfit(x, y, 1, w=1.0 / sig)[0])


stats = {}
b_all, e_all = wslope(r)
b, e = wslope(good)
stats["slope_all"] = [b_all, e_all]
stats["slope_good"] = [b, e]
stats["amplitude"] = [b / SLOPE_TH, e / abs(SLOPE_TH)]
rho_s, p_s = sps.spearmanr(good.kappa_ext, good.hr)
stats["spearman"] = [rho_s, p_s]

# ------------------------------------------------- permutation null (1.1) --
x, y, sig = (good.kappa_ext.to_numpy(), good.hr.to_numpy(),
             good.MUERR.to_numpy())
zbin = good.zbin.to_numpy()
n_perm = 10000
perm = np.empty(n_perm)
idx_by_bin = [np.flatnonzero(zbin == zb) for zb in np.unique(zbin)]
yp = y.copy()
for k in range(n_perm):
    for idx in idx_by_bin:
        yp[idx] = y[rng.permutation(idx)]
    perm[k] = slope_only(x, yp, sig)
p_perm = float(np.mean(perm <= b))          # one-sided (lensing: b < 0)
stats["permutation"] = {
    "n_perm": n_perm, "p_one_sided": max(p_perm, 1.0 / n_perm),
    "null_sigma": float(perm.std()),
    "z_equiv": float(abs(b - perm.mean()) / perm.std()),
}

# -------------------------------------- spatial block bootstrap (1.1) ------
BLOCK_DEG = 0.5   # > 2x aperture radius (10 arcmin)
bra = np.floor(good.HOST_RA.to_numpy() / BLOCK_DEG)
bdec = np.floor(good.HOST_DEC.to_numpy() / BLOCK_DEG)
block_id = pd.factorize(bra * 1000 + bdec)[0]
blocks = [np.flatnonzero(block_id == u) for u in np.unique(block_id)]
n_bboot = 4000
bb = np.empty(n_bboot)
for k in range(n_bboot):
    pick = rng.integers(0, len(blocks), len(blocks))
    idx = np.concatenate([blocks[j] for j in pick])
    bb[k] = slope_only(x[idx], y[idx], sig[idx])
stats["block_bootstrap"] = {"block_deg": BLOCK_DEG, "n_blocks": len(blocks),
                            "slope_err": float(bb.std())}

# ------------------------------- host-mass confounder tests (1.2b, 1.2c) --
hm_ok = np.isfinite(good.HOST_LOGMASS.to_numpy()) \
    & (good.HOST_LOGMASS.to_numpy() > 0)
hx = good.HOST_LOGMASS.to_numpy()[hm_ok]
rho_hm, p_hm = sps.spearmanr(hx, x[hm_ok])
stats["kappa_vs_hostmass"] = {"n": int(hm_ok.sum()),
                              "spearman_rho": float(rho_hm),
                              "p": float(p_hm)}
# slope with a host-mass-step covariate: y = b*kappa + m*step + const
step = (hx >= 10.0).astype(float)
A = np.column_stack([x[hm_ok], step, np.ones(step.size)])
W = 1.0 / sig[hm_ok] ** 2
ATA = A.T @ (A * W[:, None]); ATy = A.T @ (W * y[hm_ok])
coef = np.linalg.solve(ATA, ATy)
boot = np.empty(2000)
for k in range(2000):
    i = rng.integers(0, step.size, step.size)
    Ai = A[i]; Wi = W[i]
    boot[k] = np.linalg.solve(Ai.T @ (Ai * Wi[:, None]),
                              Ai.T @ (Wi * y[hm_ok][i]))[0]
stats["slope_with_masstep"] = {"slope": float(coef[0]),
                               "err": float(boot.std()),
                               "mass_step_mag": float(coef[1])}

# ------------------------------------------------------------- subgroups --
groups = {}
for g in ("X", "S", "C", "E"):
    d = good[good.GROUP == g]
    groups[g] = wslope(d) + (len(d),)
d = good[good.GROUP.isin(["X", "S"])]; groups["X+S"] = wslope(d) + (len(d),)
d = good[good.GROUP.isin(["C", "E"])]; groups["C+E"] = wslope(d) + (len(d),)
stats["groups"] = {k: list(v) for k, v in groups.items()}

# jackknife over field groups
jk = [wslope(good[good.GROUP != g])[0] for g in ("X", "S", "C", "E")]
stats["jackknife_range"] = [min(jk), max(jk)]

# ------------------------------------------------- robustness rows (3.6/7) --
rob = {}
d = good[good.zHD < 1.0]
rob["z_lt_1"] = list(wslope(d)) + [len(d)]
# de-lensed weights: DES's MUERR already contains sigma_lens = 0.055 z, which
# down-weights exactly the SNe carrying the signal; remove it in quadrature
sig_dl = np.sqrt(np.clip(good.MUERR.to_numpy() ** 2
                         - (0.055 * good.zHD.to_numpy()) ** 2,
                         0.05 ** 2, None))
b_dl, e_dl = bootstrap_slope(x, y, sig_dl, rng, n_boot=4000)
rob["delensed_weights"] = [b_dl, e_dl, len(good)]
if "area_flag" in good:
    d = good[~good.area_flag.astype(bool)]
    rob["area_clean"] = list(wslope(d)) + [len(d)]
for var, fn in (("excise", "des_all_kappa_excise.csv"),
                ("nospecz", "des_all_kappa_nospecz.csv"),
                ("w1only", "des_all_kappa_w1only.csv"),
                ("naive", "des_all_kappa_naive.csv"),
                ("cap14.1", "des_all_kappa_cap14.1.csv"),
                ("mstar05", "des_all_kappa_mstar05.csv")):
    p = ROOT / "output/des_full" / fn
    if p.exists():
        dv = pd.read_csv(p)
        dv = dv[dv.PROBIA > 0.9]
        rob[f"variant_{var}"] = list(wslope(dv)) + [len(dv)]
        if var == "nospecz":  # compare against headline on the same groups
            dh = good[good.GROUP.isin(dv.GROUP.unique())]
            rob["headline_XS_for_nospecz"] = list(wslope(dh)) + [len(dh)]
stats["robustness"] = rob

# ------------------------------------------------------------- means etc. --
hi = good[good.kappa_ext > 0.005]; lo = good[good.kappa_ext <= 0.005]
w_hi, w_lo = 1 / hi.MUERR**2, 1 / lo.MUERR**2
m_hi = np.average(hi.hr, weights=w_hi); m_lo = np.average(lo.hr, weights=w_lo)
e_hi = np.sqrt(1 / w_hi.sum()); e_lo = np.sqrt(1 / w_lo.sum())
stats["himean"] = [m_hi, e_hi, len(hi)]
stats["lomean"] = [m_lo, e_lo, len(lo)]
stats["hilo_sig"] = (m_lo - m_hi) / np.hypot(e_hi, e_lo)
stats["sigma_kappa"] = float(good.kappa_ext.std())
stats["n_all"], stats["n_good"] = len(r), len(good)

# ---------------------------------------------------------------- fig 1 --
fig, ax = plt.subplots(figsize=(3.5, 2.9))
xg, yg = good.kappa_ext.to_numpy(), good.hr.to_numpy()
m = (xg > -0.015) & (xg < 0.055)
ax.plot(xg[m], yg[m], ".", ms=2.2, color=GRAY, alpha=0.45, rasterized=True)
edges = np.quantile(xg, np.linspace(0.02, 0.995, 9))
xb, yb, eb = [], [], []
for a_, b_ in zip(edges[:-1], edges[1:]):
    s = (xg >= a_) & (xg < b_)
    if s.sum() < 5:
        continue
    ww = 1 / good.MUERR.to_numpy()[s] ** 2
    xb.append(np.average(xg[s], weights=ww))
    yb.append(np.average(yg[s], weights=ww))
    eb.append(np.sqrt(1 / ww.sum()))
ax.errorbar(xb, yb, yerr=eb, fmt="o", ms=4, color="k", lw=1, capsize=2,
            zorder=5)
xx = np.linspace(-0.015, 0.055, 10)
ax.plot(xx, SLOPE_TH * xx, "--", color=VERM, lw=1.6, zorder=4)
ax.plot(xx, b * xx, "-", color=BLUE, lw=1.6, zorder=4)
ax.text(0.031, SLOPE_TH * 0.052, "prediction\n$-2.17\\,\\kappa$",
        color=VERM, fontsize=8, va="top")
ax.text(0.040, b * 0.028, "best fit", color=BLUE, fontsize=8, va="bottom")
ax.set_xlim(-0.015, 0.055); ax.set_ylim(-0.42, 0.42)
ax.set_xlabel(r"predicted external convergence $\kappa_{\rm ext}$")
ax.set_ylabel(r"Hubble residual $\Delta\mu$ [mag]")
ax.axhline(0, color="k", lw=0.5, alpha=0.4)
fig.tight_layout(); fig.savefig(HERE / "fig1_slope.pdf"); plt.close(fig)

# ---------------------------------------------------------------- fig 2 --
zb = r.groupby("zbin").agg(zp=("rand_mean", "mean"),
                           sig=("rand_sig", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.plot(zb.zbin, zb.zp, "o-", color=BLUE, ms=3.5, lw=1.3)
ax.plot(zb.zbin, zb.sig, "s--", color=VERM, ms=3.5, lw=1.3)
zz = np.linspace(0.1, 1.15, 20)
ax.plot(zz, 0.055 * zz / abs(SLOPE_TH), ":", color="k", lw=1.2)
ax.text(0.52, 0.0118, r"mean $\langle\kappa\rangle$ (random LOS)",
        color=BLUE, fontsize=8, rotation=17)
ax.text(0.60, 0.0042, r"robust $\sigma_\kappa$ (random LOS)", color=VERM,
        fontsize=8, rotation=11)
ax.text(0.30, 0.0128, r"$\sigma_{\rm lens}/2.17$ ($0.055z$, N-body)",
        color="k", fontsize=8, rotation=26)
ax.set_xlabel(r"source redshift $z$")
ax.set_ylabel(r"$\kappa$")
ax.set_xlim(0.1, 1.15); ax.set_ylim(0, 0.030)
fig.tight_layout(); fig.savefig(HERE / "fig2_zeropoint.pdf"); plt.close(fig)

# ---------------------------------------------------------------- fig 3 --
order = ["X", "S", "C", "E", "X+S", "C+E"]
labels = [f"{g}  (N={groups[g][2]})" for g in order] + \
         [f"combined  (N={len(good)})"]
amps = [groups[g][0] / SLOPE_TH for g in order] + [b / SLOPE_TH]
errs = [groups[g][1] / abs(SLOPE_TH) for g in order] + [e / abs(SLOPE_TH)]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ypos = np.arange(len(labels))[::-1]
ax.axvline(0, color=GRAY, lw=1, ls=":")
ax.axvline(1, color=VERM, lw=1.4, ls="--")
ax.text(1.1, ypos[0] + 0.35, "prediction", color=VERM, fontsize=8,
        ha="left")
ax.text(-0.1, ypos[0] + 0.35, "no lensing", color=GRAY, fontsize=8,
        ha="right")
for yp_, a_, e_, lab in zip(ypos, amps, errs, labels):
    c = BLUE if "combined" in lab else "k"
    ax.errorbar(a_, yp_, xerr=e_, fmt="o", color=c,
                ms=4.5 if c == BLUE else 3.5, lw=1.4, capsize=2.5)
ax.set_yticks(ypos, labels, fontsize=8)
ax.set_xlabel(r"lensing amplitude $A \equiv b_{\rm fit}/b_{\rm pred}$")
ax.set_xlim(-1.6, 3.2)
fig.tight_layout(); fig.savefig(HERE / "fig3_forest.pdf"); plt.close(fig)

# ------------------------------------------- top sightlines table (3.2) --
top = good.nlargest(10, "kappa_ext")[
    ["CID", "FIELD", "zHD", "kappa_ext", "hr"]].copy()
# exact point-mass-free magnification (shear from the same halo sum is not
# computed; |gamma| ~ kappa for isolated halos would brighten further, so
# this is the conservative no-shear value)
top["dmu_pred"] = 5.0 * np.log10(1.0 - top.kappa_ext)
top.to_csv(HERE / "table1_top.csv", index=False)

(HERE / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
for k, v in stats.items():
    print(k, np.round(v, 4) if isinstance(v, (list, float)) else v)
