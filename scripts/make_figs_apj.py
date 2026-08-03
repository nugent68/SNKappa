#!/usr/bin/env python
"""ApJ paper figures, tables, and consolidated statistics.

Produces output/figs_apj/{f1..f7}.pdf, table CSVs, and
output/figs_apj/paper_stats_apj.json — every number the ApJ manuscript
quotes, drawn from the tracked artifacts (never recomputed from
scratch except the cheap pair matches for F5/F6).

F1 slope triptych (DES / Union3 / Pantheon+)     F5 cross-pipeline kappa
F2 zero point + sigma_kappa(z)                   F6 standardization test
F3 grand amplitude forest                        F7 DESI DR2 trio
F4 two-tier DeltaSigma closure

Run: .venv/bin/python scripts/make_figs_apj.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snkappa.kappa import angular_sep_arcsec

FIG = Path("output/figs_apj"); FIG.mkdir(parents=True, exist_ok=True)
BLUE, VERM, GRAY, GREEN = "#0072B2", "#D55E00", "#888888", "#009E73"
SLOPE_TH = -5.0 / np.log(10.0)
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "figure.dpi": 150,
})


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def load_catalogs():
    des = pd.read_csv("output/des_full/des_all_kappa.csv")
    des = des[des.PROBIA > 0.9].copy()
    u3 = pd.read_csv("output/union3/union3_kappa.csv")
    u3 = u3[~u3.clipped.astype(bool)
            & ~u3.cluster_targeted.astype(bool)].copy()
    pp = pd.read_csv("output/pantheon/pantheon_kappa.csv")
    pp = pp[~pp.clipped.astype(bool)
            & ~pp.cluster_targeted.astype(bool)].copy()
    return {"DES-SN5YR": des, "Union3": u3, "Pantheon+": pp}


def match(a, b):
    bra, bdec, bz = (b.HOST_RA.to_numpy(), b.HOST_DEC.to_numpy(),
                     b.zHD.to_numpy())
    order = np.argsort(bdec)
    bdec_s = bdec[order]
    ii, jj = [], []
    for i, r in a.reset_index(drop=True).iterrows():
        k0 = np.searchsorted(bdec_s, r.HOST_DEC - 0.001)
        k1 = np.searchsorted(bdec_s, r.HOST_DEC + 0.001)
        if k1 <= k0:
            continue
        cand = order[k0:k1]
        sep = angular_sep_arcsec(r.HOST_RA, r.HOST_DEC, bra[cand],
                                 bdec[cand])
        j = np.argmin(sep)
        if sep[j] < 1.5 and abs(bz[cand[j]] - r.zHD) < 0.01:
            ii.append(i); jj.append(cand[j])
    return np.array(ii), np.array(jj)


def f1_triptych(cats, stats):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    slopes = {"DES-SN5YR": stats["des"]["slope_good"],
              "Union3": stats["union3"]["slope_good"],
              "Pantheon+": stats["pantheon"]["slope_good"]}
    for ax, (name, d) in zip(axes, cats.items()):
        x, y = d.kappa_ext.to_numpy(), d.hr.to_numpy()
        m = (x > -0.015) & (x < 0.055)
        ax.plot(x[m], y[m], ".", ms=1.8, color=GRAY, alpha=0.4,
                rasterized=True)
        edges = np.quantile(x, np.linspace(0.02, 0.995, 8))
        for a_, b_ in zip(edges[:-1], edges[1:]):
            s = (x >= a_) & (x < b_)
            if s.sum() < 5:
                continue
            w = 1 / d.MUERR.to_numpy()[s] ** 2
            ax.errorbar(np.average(x[s], weights=w),
                        np.average(y[s], weights=w),
                        yerr=np.sqrt(1 / w.sum()), fmt="o", ms=3.5,
                        color="k", lw=1, capsize=2, zorder=5)
        xx = np.linspace(-0.015, 0.055, 10)
        ax.plot(xx, SLOPE_TH * xx, "--", color=VERM, lw=1.4)
        ax.plot(xx, slopes[name][0] * xx, "-", color=BLUE, lw=1.4)
        ax.set_xlim(-0.015, 0.055); ax.set_ylim(-0.42, 0.42)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_title(f"{name} (N={len(d)})", fontsize=9)
        ax.set_xlabel(r"$\kappa_{\rm ext}$")
    axes[0].set_ylabel(r"$\Delta\mu$ [mag]")
    fig.tight_layout(); fig.savefig(FIG / "f1_slopes.pdf")
    plt.close(fig)


def f2_zeropoint():
    r = pd.read_csv("output/des_full/des_all_kappa.csv")
    zb = r.groupby("zbin").agg(zp=("rand_mean", "mean"),
                               sig=("rand_sig", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(zb.zbin, zb.zp, "o-", color=BLUE, ms=3.5, lw=1.3,
            label=r"mean $\langle\kappa\rangle$ (random LOS)")
    ax.plot(zb.zbin, zb.sig, "s--", color=VERM, ms=3.5, lw=1.3,
            label=r"robust $\sigma_\kappa$ (random LOS)")
    zz = np.linspace(0.1, 1.15, 20)
    ax.plot(zz, 0.055 * zz / abs(SLOPE_TH), ":", color="k", lw=1.2,
            label=r"$\sigma_{\rm lens}/2.17$ ($0.055z$)")
    ax.set_xlabel(r"source redshift $z$"); ax.set_ylabel(r"$\kappa$")
    ax.set_xlim(0.1, 1.15); ax.set_ylim(0, 0.030)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "f2_zeropoint.pdf")
    plt.close(fig)


def f3_forest(stats):
    rows = []      # (label, A, err, class)
    S = abs(SLOPE_TH)

    def add(lbl, sl, cls):
        rows.append((lbl, sl[0] / SLOPE_TH, sl[1] / S, cls))

    ds = stats["des"]
    for g in ("X", "S", "C", "E"):
        add(f"DES {g}", ds["groups"][g][:2], "sub")
    add("DES-SN5YR", ds["slope_good"], "main")
    u3 = stats["union3"]
    for s in ("SDSS", "Pan-STARRS", "SNLS", "ESSENCE"):
        k = f"slope_{s}"
        if k in u3:
            add(f"U3 {s}", u3[k][:2], "sub")
    add("Union3", u3["slope_good"], "main")
    pp = stats["pantheon"]
    for s in ("SDSS", "PS1MD", "DES", "SNLS"):
        k = f"slope_{s}"
        if k in pp:
            add(f"P+ {s}", pp[k][:2], "sub")
    add("Pantheon+", pp["slope_good"], "main")
    g = stats["pantheon_gls"]
    add("P+ GLS full cov", g["slope_gls_fullcov"], "method")
    for name, key in (("des", "DES"), ("union3", "U3"),
                      ("pantheon", "P+")):
        lf = stats.get(f"latent_{name}")
        if lf:
            rows.append((f"{key} latent EIV", lf["single"]["A"],
                         lf["single"]["A_err"], "method"))
    j = stats["joint"]
    rows.append(("JOINT (2920 sightlines)", j["A_joint"][0],
                 j["A_joint"][1], "joint"))

    fig, ax = plt.subplots(figsize=(4.2, 0.28 * len(rows) + 1.2))
    ypos = np.arange(len(rows))[::-1]
    ax.axvline(0, color=GRAY, lw=1, ls=":")
    ax.axvline(1, color=VERM, lw=1.4, ls="--")
    colors = {"sub": "k", "main": BLUE, "method": GREEN, "joint": VERM}
    for yp, (lbl, a, e, cls) in zip(ypos, rows):
        ax.errorbar(a, yp, xerr=e, fmt="o", color=colors[cls],
                    ms=4.5 if cls in ("main", "joint") else 3.2,
                    lw=1.3, capsize=2.2)
    ax.set_yticks(ypos, [r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel(r"lensing amplitude $A$")
    ax.set_xlim(-1.6, 3.4)
    fig.tight_layout(); fig.savefig(FIG / "f3_forest.pdf")
    plt.close(fig)
    return rows


def f4_closure():
    boss = jload("output/delta_sigma/boss_closure_recal.json")
    boss0 = jload("output/delta_sigma/boss_closure_nir1um.json")
    desi = jload("output/delta_sigma/desi_closure_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for i, b in enumerate(boss["bins"]):
        rp = np.array(b["rp"]); c = plt.cm.viridis(i / 3)
        ax.errorbar(rp, np.array(b["ds_meas"]), yerr=np.array(b["ds_err"]),
                    fmt="o", ms=2.5, color=c, lw=0.8, capsize=1.5,
                    label=f"{b['sample']} {b['zmin']}-{b['zmax']}")
        ax.plot(rp, np.array(b["ds_pred"]), "-", color=c, lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_p$ [Mpc]")
    ax.set_ylabel(r"$\Delta\Sigma$ [$M_\odot$/pc$^2$]")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("BOSS × KiDS-Legacy (public)", fontsize=8)
    ax = axes[1]
    pts = []
    for b in boss["bins"]:
        w = b["windows"]["one_halo"]
        pts.append((b["logms_med"], w["A_ds"], w["A_ds_err"], "BOSS"))
    for b in boss0["bins"]:
        w = b["windows"]["one_halo"]
        pts.append((b["logms_med"], w["A_ds"], w["A_ds_err"], "BOSS-pre"))
    for b in desi["fiducial_nir1um_fsf"]:
        for sv, (a, e) in b["A_1h"].items():
            pts.append((b["logms_med"], a, e, "DESI"))
    style = {"BOSS": (BLUE, "o"), "BOSS-pre": (GRAY, "x"),
             "DESI": (VERM, "s")}
    seen = set()
    for m, a, e, cls in pts:
        c, mk = style[cls]
        lbl = {"BOSS": "BOSS×KiDS (fiducial)",
               "BOSS-pre": "constant $M_*/L$",
               "DESI": "DESI DR1 (LWB)"}[cls] if cls not in seen else None
        seen.add(cls)
        ax.errorbar(m, a, yerr=e, fmt=mk, ms=3.5, color=c, lw=1,
                    capsize=1.5, label=lbl, alpha=0.85)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"median $\log M_\star$")
    ax.set_ylabel(r"$A_{1h}$ = meas/pred")
    ax.set_ylim(0.4, 2.4)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("one-halo closure vs mass", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "f4_closure.pdf")
    plt.close(fig)


def f5_f6_pairs(cats):
    u3, pp = cats["Union3"], cats["Pantheon+"]
    des = cats["DES-SN5YR"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    i1, j1 = match(u3[u3["sample"].str.startswith("DES3")], des)
    a = u3[u3["sample"].str.startswith("DES3")].reset_index(drop=True)
    ax.plot(des.kappa_ext.to_numpy()[j1], a.kappa_ext.to_numpy()[i1],
            "o", ms=3, color=BLUE, alpha=0.7,
            label=f"U3(DES3)×DES  r={np.corrcoef(des.kappa_ext.to_numpy()[j1], a.kappa_ext.to_numpy()[i1])[0,1]:.3f}")
    i2, j2 = match(pp, u3)
    ppr = pp.reset_index(drop=True)
    r2 = np.corrcoef(ppr.kappa_ext.to_numpy()[i2],
                     u3.kappa_ext.to_numpy()[j2])[0, 1]
    ax.plot(u3.kappa_ext.to_numpy()[j2], ppr.kappa_ext.to_numpy()[i2],
            "s", ms=2, color=VERM, alpha=0.5,
            label=f"P+×U3  r={r2:.3f}")
    lim = (-0.02, 0.09)
    ax.plot(lim, lim, "-", color=GRAY, lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\kappa_{\rm ext}$ (pipeline A)")
    ax.set_ylabel(r"$\kappa_{\rm ext}$ (pipeline B)")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    kk = 0.5 * (ppr.kappa_ext.to_numpy()[i2] + u3.kappa_ext.to_numpy()[j2])
    dhr = ppr.hr.to_numpy()[i2] - u3.hr.to_numpy()[j2]
    ax.plot(kk, dhr, ".", ms=3, color=GRAY, alpha=0.6, rasterized=True)
    w = 1.0 / np.hypot(ppr.MUERR.to_numpy()[i2], u3.MUERR.to_numpy()[j2])
    bd = np.polyfit(kk, dhr, 1, w=w)
    xx = np.linspace(-0.01, 0.05, 10)
    ax.plot(xx, np.polyval(bd, xx), "-", color=BLUE, lw=1.4,
            label=f"slope = {bd[0]:+.2f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\kappa_{\rm ext}$ (matched)")
    ax.set_ylabel(r"$\Delta\mu_{\rm P+} - \Delta\mu_{\rm U3}$ [mag]")
    ax.set_ylim(-0.6, 0.6)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "f5_crosspipe.pdf")
    plt.close(fig)
    return float(np.corrcoef(des.kappa_ext.to_numpy()[j1],
                             a.kappa_ext.to_numpy()[i1])[0, 1]), r2


def f7_trio():
    des = jload("output/des_full/desi_impact.json")
    dd_des = [s["ddchi2"] for s in des["scan"] if abs(s["A"] - 0.79) < .01][0]
    u3 = jload("output/union3/desi_leg.json")["ddchi2"]
    pp = jload("output/pantheon/gls_fit.json")["desi_pantheon_leg"]["ddchi2"]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    labels = ["DES-SN5YR\n(4.2$\\sigma$ leg)", "Union3\n(3.8$\\sigma$)",
              "Pantheon+\n(2.8$\\sigma$)"]
    vals = [dd_des, u3, pp]
    ax.bar(labels, vals, color=[BLUE, VERM, GREEN], width=0.55)
    ax.axhline(7.7, color="k", ls="--", lw=1)
    ax.text(0.02, 7.9, r"$\Delta\Delta\chi^2$ needed for $5\sigma$ (DES leg)",
            fontsize=7)
    ax.set_ylabel(r"$\Delta\Delta\chi^2$ from de-lensing")
    ax.set_ylim(-1, 9)
    ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout(); fig.savefig(FIG / "f7_desi_trio.pdf")
    plt.close(fig)
    return {"DES": dd_des, "Union3": u3, "PantheonP": pp}


def main():
    cats = load_catalogs()
    stats = {
        "des": jload("output/des_full/paper_stats.json"),
        "union3": jload("output/union3/fit_summary.json"),
        "pantheon": jload("output/pantheon/fit_summary.json"),
        "pantheon_gls": jload("output/pantheon/gls_fit.json"),
        "joint": jload("output/joint/joint_fit.json"),
        "latent_des": jload("output/des_full/latent_fit.json"),
        "latent_union3": jload("output/union3/latent_fit.json"),
        "latent_pantheon": jload("output/pantheon/latent_fit.json"),
        "union3_extra": jload("output/union3/extra_stats.json"),
        "pantheon_extra": jload("output/pantheon/extra_stats.json"),
        "boss_closure": jload("output/delta_sigma/boss_closure_recal.json"),
        "desi_closure": jload("output/delta_sigma/desi_closure_summary.json"),
        "delens": jload("output/des_full/delens_demo.json"),
        "moments": jload("output/des_full/moments_test.json"),
        "latent_rob": jload("output/des_full/latent_robustness.json"),
    }
    f1_triptych(cats, stats)
    f2_zeropoint()
    forest = f3_forest(stats)
    f4_closure()
    r_du, r_pu = f5_f6_pairs(cats)
    trio = f7_trio()

    # T1: top-15 sightlines across compilations (unique by position)
    allc = pd.concat([c.assign(compilation=n) for n, c in cats.items()])
    top = allc.nlargest(40, "kappa_ext").drop_duplicates(
        subset="CID").nlargest(15, "kappa_ext")
    top[["CID", "compilation", "zHD", "kappa_ext", "gamma", "dmu_pred",
         "hr"]].to_csv(FIG / "t1_top15.csv", index=False)

    stats["cross_pipeline_corr"] = {"U3(DES3)xDES": r_du, "P+xU3": r_pu}
    stats["desi_trio_ddchi2"] = trio
    stats["forest_rows"] = [{"label": r[0], "A": r[1], "err": r[2]}
                            for r in forest]
    (FIG / "paper_stats_apj.json").write_text(
        json.dumps(stats, indent=1, default=float))
    print(f"figures + tables + paper_stats_apj.json in {FIG}")
    missing = [k for k, v in stats.items() if v is None]
    if missing:
        print("MISSING artifacts (rerun after latent chain):", missing)


if __name__ == "__main__":
    main()
