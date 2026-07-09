#!/usr/bin/env python
"""Paper statistics + figures for the ApJL manuscript.

Reads SNKappa/output/des_full/des_all_kappa.csv; writes fig1-3.pdf and
stats.json into this directory. Palette: Okabe-Ito blue/vermilion
(CVD-validated); identity always carried by linestyle + direct label too.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).parent
CSV = HERE.parent / "SNKappa/output/des_full/des_all_kappa.csv"
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
    x, y, w = d.kappa_ext.to_numpy(), d.hr.to_numpy(), 1 / d.MUERR.to_numpy() ** 2
    b = np.polyfit(x, y, 1, w=w)[0]
    boot = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        boot[k] = np.polyfit(x[i], y[i], 1, w=w[i])[0]
    return b, boot.std()


stats = {}
b_all, e_all = wslope(r)
b, e = wslope(good)
stats["slope_all"] = [b_all, e_all]
stats["slope_good"] = [b, e]
stats["amplitude"] = [b / SLOPE_TH, e / abs(SLOPE_TH)]
rho_s, p_s = sps.spearmanr(good.kappa_ext, good.hr)
stats["spearman"] = [rho_s, p_s]

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
x, y = good.kappa_ext.to_numpy(), good.hr.to_numpy()
m = (x > -0.015) & (x < 0.055)
ax.plot(x[m], y[m], ".", ms=2.2, color=GRAY, alpha=0.45, rasterized=True)
edges = np.quantile(x, np.linspace(0.02, 0.995, 9))
xb, yb, eb = [], [], []
for a_, b_ in zip(edges[:-1], edges[1:]):
    s = (x >= a_) & (x < b_)
    if s.sum() < 5:
        continue
    ww = 1 / good.MUERR.to_numpy()[s] ** 2
    xb.append(np.average(x[s], weights=ww))
    yb.append(np.average(y[s], weights=ww))
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
for yp, a_, e_, lab in zip(ypos, amps, errs, labels):
    c = BLUE if "combined" in lab else "k"
    ax.errorbar(a_, yp, xerr=e_, fmt="o", color=c, ms=4.5 if c == BLUE else 3.5,
                lw=1.4, capsize=2.5)
ax.set_yticks(ypos, labels, fontsize=8)
ax.set_xlabel(r"lensing amplitude $A \equiv b_{\rm fit}/b_{\rm pred}$")
ax.set_xlim(-1.6, 3.2)
fig.tight_layout(); fig.savefig(HERE / "fig3_forest.pdf"); plt.close(fig)

# top sightlines table data
top = good.nlargest(10, "kappa_ext")[
    ["CID", "FIELD", "zHD", "kappa_ext", "hr"]]
top.to_csv(HERE / "table1_top.csv", index=False)

(HERE / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
for k, v in stats.items():
    print(k, np.round(v, 4) if isinstance(v, (list, float)) else v)
