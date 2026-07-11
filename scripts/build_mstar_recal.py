#!/usr/bin/env python
"""Build the Nir1um -> FastSpecFit stellar-mass recalibration table.

Motivated by the DeltaSigma closure (output/delta_sigma/): the constant
M*/L_1um = 0.6 of the Nir1um estimator runs 0.2-0.25 dex low for massive
red galaxies (old stellar populations have higher rest-1um M/L), which the
steep SMHM inversion amplifies into a ~x2 deficit in predicted lensing.

Calibration sample: DESI DR1 BGS_BRIGHT (z 0.05-0.45; blue+red) and LRG
(z 0.4-1.1; red) LSS catalogs, whose own LS photometry feeds our estimator,
matched by TARGETID to the PUBLIC FastSpecFit VAC (iron v3.0) LOGMSTAR.
The correction is the median residual Delta = logM*_FSF - logM*_Nir1um in
bins of (logM*_Nir1um, z), nearest-filled where unsampled, written to
snkappa/data/nir1um_fsf_recal.json for the Nir1umFSF estimator.

Also writes a diagnostic of the residual against observed z-W1 color to
document that the (logM*, z) parameterization captures the trend.

Run on Perlmutter: ./venv/bin/python scripts/build_mstar_recal.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa.stellar import make_estimator
from delta_sigma_closure import CATDIR, mags_from_fluxes
from mstar_vs_fastspecfit import load_fsf_logmstar

LOGM_EDGES = np.arange(8.0, 12.26, 0.25)
Z_EDGES = np.arange(0.05, 1.16, 0.10)
MIN_PER_CELL = 50


def collect(sample, zlo, zhi, program, est, rng, n_max=250000):
    t = Table.read(CATDIR / f"{sample}_clustering.dat.fits")
    sel = (np.asarray(t["Z"]) > zlo) & (np.asarray(t["Z"]) < zhi)
    idx = np.flatnonzero(sel)
    if idx.size > n_max:
        idx = rng.choice(idx, n_max, replace=False)
    t = t[idx]
    z = np.asarray(t["Z"], dtype=float)
    mags = mags_from_fluxes(t, sample)
    ours = est.logmstar(mags, z)
    tid_f, lm_f = load_fsf_logmstar(program)
    order = np.argsort(tid_f)
    tid = np.asarray(t["TARGETID"])
    pos = np.clip(np.searchsorted(tid_f[order], tid), 0, order.size - 1)
    row = order[pos]
    good = (tid_f[row] == tid) & np.isfinite(ours)
    fsf = lm_f[row]
    good &= np.isfinite(fsf) & (fsf > 6)
    zw1 = mags["z"] - mags["w1"]
    return (z[good], ours[good], fsf[good] - ours[good], zw1[good])


def main():
    rng = np.random.default_rng(20130901)
    from astropy.cosmology import Planck18 as cosmo
    est = make_estimator("nir1um", cosmo)

    zs, ms, dm, zw1 = [], [], [], []
    for sample, zlo, zhi, prog in (("BGS_BRIGHT", 0.05, 0.45, "bright"),
                                   ("LRG", 0.40, 1.10, "dark")):
        z, m, d, c = collect(sample, zlo, zhi, prog, est, rng)
        print(f"{sample}: {z.size} matched (median resid {np.median(d):+.3f})",
              flush=True)
        zs.append(z); ms.append(m); dm.append(d); zw1.append(c)
    z = np.concatenate(zs); m = np.concatenate(ms)
    d = np.concatenate(dm); c = np.concatenate(zw1)

    # ---- parametric fit -----------------------------------------------
    # Binning the residual against the RAW estimate suffers Eddington bias
    # at each sample's selection edge (raw estimates that scattered low at
    # fixed true mass produce spuriously large medians there). Instead fit
    # a smooth monotone model against the FSF mass,
    #     Delta(m, z) = c0 + A(z) * sigmoid((m - m0)/w),
    # with A(z) piecewise-linear on z nodes; the estimator applies it via
    # fixed-point iteration on the raw estimate.
    m_fsf = m + d
    Z_NODES = np.array([0.10, 0.30, 0.50, 0.70, 0.90, 1.05])

    def model(p, mm, zz):
        c0, m0, w = p[0], p[1], p[2]
        a = np.interp(zz, Z_NODES, p[3:])
        return c0 + a / (1.0 + np.exp(-(mm - m0) / np.clip(w, 0.05, 2.0)))

    from scipy.optimize import least_squares
    sub = rng.choice(z.size, min(200000, z.size), replace=False)
    p0 = np.concatenate([[-0.03, 10.5, 0.3], np.full(Z_NODES.size, 0.3)])

    def resid(p):
        return model(p, m_fsf[sub], z[sub]) - d[sub]

    # bounds keep the parameterization identifiable (an unbounded fit finds
    # c0 ~ -1500, A ~ +1500 with the sigmoid nearly saturated -- a locally
    # good but catastrophically extrapolating solution)
    lo = np.concatenate([[-0.15, 9.5, 0.10], np.zeros(Z_NODES.size)])
    hi = np.concatenate([[+0.15, 11.5, 1.00], np.full(Z_NODES.size, 0.6)])
    fit = least_squares(resid, p0, loss="soft_l1", f_scale=0.15,
                        bounds=(lo, hi))
    p = fit.x
    print("fit params: c0=%.3f m0=%.2f w=%.2f A(z)=%s"
          % (p[0], p[1], p[2], np.round(p[3:], 3).tolist()), flush=True)

    # diagnostics: median residual after correction, per (z, FSF-mass) cell
    nz, nm = Z_EDGES.size - 1, LOGM_EDGES.size - 1
    med = np.full((nz, nm), np.nan)
    cnt = np.zeros((nz, nm), int)
    dcorr = d - model(p, m_fsf, z)
    iz = np.clip(np.digitize(z, Z_EDGES) - 1, 0, nz - 1)
    im = np.clip(np.digitize(m_fsf, LOGM_EDGES) - 1, 0, nm - 1)
    for i in range(nz):
        for j in range(nm):
            s = (iz == i) & (im == j)
            cnt[i, j] = s.sum()
            if cnt[i, j] >= MIN_PER_CELL:
                med[i, j] = np.median(dcorr[s])

    # color diagnostic: residual vs observed z-W1 in coarse z slices
    diag = {}
    for zlo, zhi in ((0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.1)):
        s = (z >= zlo) & (z < zhi) & np.isfinite(c)
        if s.sum() < 500:
            continue
        q = np.quantile(c[s], [0.1, 0.3, 0.5, 0.7, 0.9])
        rows = []
        for clo, chi in zip(q[:-1], q[1:]):
            ss = s & (c >= clo) & (c < chi)
            rows.append([float(0.5 * (clo + chi)),
                         float(np.median(d[ss])), int(ss.sum())])
        diag[f"z{zlo}-{zhi}"] = rows

    out = {
        "description": ("Delta(m,z) = c0 + A(z) sigmoid((m-m0)/w) with m on "
                        "the FSF mass scale; apply by fixed point on the "
                        "raw Nir1um estimate"),
        "model": "sigmoid",
        "c0": float(p[0]), "m0": float(p[1]), "w": float(p[2]),
        "z_nodes": Z_NODES.tolist(),
        "a_nodes": p[3:].tolist(),
        "n_total": int(z.size),
        "resid_nmad_after": float(1.4826 * np.median(np.abs(
            dcorr - np.median(dcorr)))),
        "color_diagnostic_zw1": diag,
        "provenance": "DESI DR1 LSS BGS_BRIGHT+LRG x FastSpecFit iron v3.0",
    }
    pth = Path("snkappa/data"); pth.mkdir(exist_ok=True)
    (pth / "nir1um_fsf_recal.json").write_text(json.dumps(out, indent=1))
    with np.printoptions(precision=2, suppress=True, linewidth=200):
        print("post-fit residual medians (rows=z, cols=FSF logM*):")
        print(np.nan_to_num(med))
    print("saved snkappa/data/nir1um_fsf_recal.json")


if __name__ == "__main__":
    main()
