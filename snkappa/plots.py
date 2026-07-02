"""Diagnostic plots (all saved as PNG into the output directory)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sky_map(outdir: Path, cfg, idx, theta, kappa_i, df):
    """kappa_i sky map of the SN aperture."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ra0, dec0 = cfg.source.ra_src, cfg.source.dec_src
    dra = (df["ra"].to_numpy()[idx] - ra0) * np.cos(np.radians(dec0)) * 3600
    ddec = (df["dec"].to_numpy()[idx] - dec0) * 3600
    k = np.clip(kappa_i, 1e-8, None)
    sc = ax.scatter(dra, ddec, c=np.log10(k), s=8, cmap="viridis")
    is_spec = df["z_spec"].notna().to_numpy()[idx]
    ax.scatter(dra[is_spec], ddec[is_spec], facecolors="none",
               edgecolors="r", s=30, linewidths=0.5, label="DESI spec-z")
    ax.plot(0, 0, "r*", ms=14, label=cfg.source.name)
    ax.set_xlabel(r"$\Delta\alpha\cos\delta$ [arcsec]")
    ax.set_ylabel(r"$\Delta\delta$ [arcsec]")
    ax.invert_xaxis()
    ax.set_title(rf"per-galaxy $\log_{{10}}\kappa_i$, {cfg.source.name}")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}\kappa_i$")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "kappa_sky_map.png", dpi=150)
    plt.close(fig)


def cumulative_profile(outdir: Path, cfg, theta, kappa_i):
    order = np.argsort(theta)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(theta[order] / 60.0, np.cumsum(kappa_i[order]), lw=2)
    ax.set_xlabel("aperture radius [arcmin]")
    ax.set_ylabel(r"cumulative $\kappa_{\rm raw}(<\theta)$")
    ax.set_title("Convergence build-up with aperture radius (fiducial)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "kappa_cumulative.png", dpi=150)
    plt.close(fig)


def pkappa_hist(outdir: Path, samples_nogroup, samples_group):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(
        min(samples_nogroup.min(), samples_group.min()),
        max(samples_nogroup.max(), samples_group.max()), 80)
    ax.hist(samples_nogroup, bins=bins, density=True, alpha=0.6,
            label="lens group excluded (default)")
    ax.hist(samples_group, bins=bins, density=True, alpha=0.6,
            label="lens group included")
    for s, c in ((samples_nogroup, "C0"), (samples_group, "C1")):
        ax.axvline(np.median(s), color=c, ls="--", lw=1)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel(r"$\kappa_{\rm ext}$")
    ax.set_ylabel(r"$P(\kappa_{\rm ext})$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "pkappa_ext.png", dpi=150)
    plt.close(fig)


def randoms_hist(outdir: Path, randoms, kappa_raw_sn):
    fig, ax = plt.subplots(figsize=(7, 5))
    k = randoms["kappa_raw"][randoms["ok"]]
    ax.hist(k, bins=50, density=True, alpha=0.7, label="random sightlines")
    ax.axvline(randoms["kappa_mean"], color="k", ls=":",
               label=r"$\langle\kappa_{\rm random}\rangle$ (zero point)")
    ax.axvline(kappa_raw_sn, color="r", lw=2, label="SN sightline (fiducial)")
    ax.set_xlabel(r"$\kappa_{\rm raw}$")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "kappa_randoms.png", dpi=150)
    plt.close(fig)


def zeta_plot(outdir: Path, cfg, randoms, zeta):
    fig, axes = plt.subplots(1, len(cfg.los.count_aperture_arcsec),
                             figsize=(6 * len(cfg.los.count_aperture_arcsec), 4.5),
                             squeeze=False)
    for ax, r_c in zip(axes[0], cfg.los.count_aperture_arcsec):
        counts = randoms["counts"][r_c][randoms["ok"]][:, 0]
        ax.hist(counts, bins=np.arange(-0.5, counts.max() + 1.5),
                density=True, alpha=0.7, label="randoms")
        sn_val = zeta[f"r_{r_c:g}_arcsec"]["unweighted"]["sn"]
        zeta_val = zeta[f"r_{r_c:g}_arcsec"]["unweighted"]["zeta"]
        ax.axvline(sn_val, color="r", lw=2,
                   label=rf"SN LOS ($\zeta$={zeta_val:.2f})")
        ax.set_xlabel(f"galaxy counts < {r_c:g}\"")
        ax.set_ylabel("density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "zeta_counts.png", dpi=150)
    plt.close(fig)
