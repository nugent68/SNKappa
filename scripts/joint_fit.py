#!/usr/bin/env python
"""Unified three-compilation lensing-amplitude fit (ApJ paper, Phase 1a).

Stacks the DES-SN5YR, Union3, and Pantheon+ kappa catalogs into one GLS
fit with an overlap-aware covariance: shared physical sightlines
(position match < 1.5", |dz| < 0.01) get cross-compilation residual
covariance rho_AB sigma_A sigma_B, with rho_AB measured EMPIRICALLY from
the matched pairs themselves. Per-compilation intercepts absorb the
independent zero points; the single shared slope is the amplitude.

Also: joint two-component (A_gal, A_cl), joint exact-dmu fit, a
dedup-representative WLS cross-check, and a permutation null (residuals
shuffled within compilation x z-bin, reusing one Cholesky factor).

Run: .venv/bin/python scripts/joint_fit.py
Output: output/joint/joint_fit.json (tracked)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.fitting import bootstrap_slope, gls_coeffs
from snkappa.kappa import angular_sep_arcsec

OUT = Path("output/joint/joint_fit.json")
SLOPE_TH = -5.0 / np.log(10.0)
N_PERM = 2000

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def load_all():
    frames = []
    d = pd.read_csv("output/des_full/des_all_kappa.csv")
    d = d[d.PROBIA > 0.9].copy()
    d["compilation"] = "DES"
    frames.append(d)
    for name, path in (("Union3", "output/union3/union3_kappa.csv"),
                       ("PantheonP", "output/pantheon/pantheon_kappa.csv")):
        t = pd.read_csv(path)
        t = t[~t.clipped.astype(bool)
              & ~t.cluster_targeted.astype(bool)].copy()
        t["compilation"] = name
        frames.append(t)
    cols = ["CID", "compilation", "HOST_RA", "HOST_DEC", "zHD", "MUERR",
            "hr", "kappa_ext", "kappa_gal_ext", "kappa_cl_ext",
            "dmu_pred", "zbin"]
    return pd.concat([f[cols] for f in frames], ignore_index=True)


def match_pairs(d, ca, cb):
    """Row-index pairs of shared sightlines between two compilations."""
    a = d[d.compilation == ca]
    b = d[d.compilation == cb]
    bra, bdec, bz = (b.HOST_RA.to_numpy(), b.HOST_DEC.to_numpy(),
                     b.zHD.to_numpy())
    order = np.argsort(bdec)
    bdec_s = bdec[order]
    pairs = []
    for i, r in a.iterrows():
        k0 = np.searchsorted(bdec_s, r.HOST_DEC - 0.001)
        k1 = np.searchsorted(bdec_s, r.HOST_DEC + 0.001)
        if k1 <= k0:
            continue
        cand = order[k0:k1]
        sep = angular_sep_arcsec(r.HOST_RA, r.HOST_DEC, bra[cand],
                                 bdec[cand])
        j = np.argmin(sep)
        if sep[j] < 1.5 and abs(bz[cand[j]] - r.zHD) < 0.01:
            pairs.append((i, b.index[cand[j]]))
    return pairs


def main():
    d = load_all().reset_index(drop=True)
    n = len(d)
    log(f"stacked rows: {n} "
        f"({dict(d.groupby('compilation').size())})")

    # ---- shared sightlines + empirical residual correlations -----------
    pair_sets = {}
    rho = {}
    for ca, cb in (("DES", "Union3"), ("Union3", "PantheonP"),
                   ("DES", "PantheonP")):
        pairs = match_pairs(d, ca, cb)
        pair_sets[(ca, cb)] = pairs
        if len(pairs) > 20:
            ii, jj = np.array(pairs).T
            r = float(np.corrcoef(d.hr.values[ii], d.hr.values[jj])[0, 1])
        else:
            r = 0.0
        rho[(ca, cb)] = r
        log(f"{ca} x {cb}: {len(pairs)} shared sightlines, "
            f"residual corr = {r:.3f}")

    # unique physical sightlines (union-find over pair graph)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pairs in pair_sets.values():
        for i, j in pairs:
            parent[find(i)] = find(j)
    comp = np.array([find(i) for i in range(n)])
    n_unique = len(np.unique(comp))
    log(f"unique physical sightlines: {n_unique}")

    # ---- overlap-aware covariance --------------------------------------
    sig = d.MUERR.to_numpy()
    C = np.diag(sig**2)
    for (ca, cb), pairs in pair_sets.items():
        r = rho[(ca, cb)]
        for i, j in pairs:
            C[i, j] = C[j, i] = r * sig[i] * sig[j]
    # ---- PRIMARY: maximal-independent (dedup) WLS ----------------------
    # At rho = 0.7-0.9 a duplicate row carries almost no independent
    # information, and a stacked GLS is fragile: the pair-DIFFERENCE
    # modes get tiny assigned variance and can dominate the fit with
    # noise (observed: a 90-sigma artifact). The defensible primary fit
    # keeps one row per physical sightline -- the best-measured one.
    rng = np.random.default_rng(1123)
    rep_idx = []
    for u in np.unique(comp):
        m = np.flatnonzero(comp == u)
        rep_idx.append(m[np.argmin(sig[m])])
    rep = d.loc[rep_idx]
    br, er = bootstrap_slope(rep.kappa_ext.to_numpy(), rep.hr.to_numpy(),
                             rep.MUERR.to_numpy(), rng)
    log(f"JOINT (dedup, best-measured rep): slope {br:+.3f} +- {er:.3f} "
        f"({abs(br)/er:.1f} sig)  A = {br/SLOPE_TH:.3f} "
        f"+- {er/abs(SLOPE_TH):.3f}  (N = {len(rep)})")
    b, e = br, er

    bx, ex = bootstrap_slope(rep.dmu_pred.to_numpy(), rep.hr.to_numpy(),
                             rep.MUERR.to_numpy(), rng)
    log(f"joint exact-dmu (dedup): A = {bx:+.3f} +- {ex:.3f}")
    from snkappa.fitting import two_component_slopes
    (bg, eg), (bc, ec) = two_component_slopes(
        rep.kappa_gal_ext.to_numpy(), rep.kappa_cl_ext.to_numpy(),
        rep.hr.to_numpy(), rep.MUERR.to_numpy(), rng)
    log(f"joint two-component (dedup): A_gal = {bg/SLOPE_TH:.2f} "
        f"+- {eg/abs(SLOPE_TH):.2f} | A_cl = {bc/SLOPE_TH:.2f} "
        f"+- {ec/abs(SLOPE_TH):.2f}")

    # representative-choice sensitivity: random representative per group
    alts = []
    for s in range(20):
        r2 = np.random.default_rng(s)
        idx2 = [np.flatnonzero(comp == u)[
            r2.integers(0, (comp == u).sum())] for u in np.unique(comp)]
        rp = d.loc[idx2]
        alts.append(np.polyfit(rp.kappa_ext, rp.hr, 1,
                               w=1.0 / rp.MUERR)[0])
    log(f"representative sensitivity: slope spread {np.std(alts):.3f} "
        f"over 20 random choices")

    # ---- consistency: stacked GLS with difference-variance floor -------
    # Cap the pair covariance so each duplicate pair's difference mode
    # keeps >= (0.06 mag)^2 variance; prevents the noise-dominated
    # difference directions from dominating.
    FLOOR2 = 0.06**2
    for (ca, cb), pairs in pair_sets.items():
        for i, j in pairs:
            cap = 0.5 * (sig[i]**2 + sig[j]**2 - FLOOR2)
            C[i, j] = C[j, i] = min(C[i, j], cap)
    try:
        cf = sla.cho_factor(C, lower=True)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        w = np.clip(w, FLOOR2 / 4, None)
        C = (V * w) @ V.T
        cf = sla.cho_factor(C, lower=True)
    comps = ("DES", "Union3", "PantheonP")
    ones = {c: (d.compilation == c).to_numpy(float) for c in comps}
    X1 = np.column_stack([d.kappa_ext.to_numpy()]
                         + [ones[c] for c in comps])
    beta_g, cov_g = gls_coeffs(X1, d.hr.to_numpy(), C, cho=cf)
    bgls, egls = float(beta_g[0]), float(np.sqrt(cov_g[0, 0]))
    log(f"stacked GLS (floored pair cov, consistency): "
        f"slope {bgls:+.3f} +- {egls:.3f}")

    # permutation null on the dedup set (compilation x zbin cells)
    key = (rep.compilation + "|" + rep.zbin.round(3).astype(str)).to_numpy()
    xr = rep.kappa_ext.to_numpy()
    yr = rep.hr.to_numpy()
    wr = 1.0 / rep.MUERR.to_numpy()
    groups = [np.flatnonzero(key == k) for k in np.unique(key)]
    perm = np.empty(N_PERM)
    yp = yr.copy()
    for k in range(N_PERM):
        for g in groups:
            yp[g] = yr[rng.permutation(g)]
        perm[k] = np.polyfit(xr, yp, 1, w=wr)[0]
    p_perm = max(float(np.mean(perm <= b)), 1.0 / N_PERM)
    z_perm = float(abs(b - perm.mean()) / perm.std())
    log(f"permutation (dedup): p = {p_perm:.2e}, "
        f"null sigma = {perm.std():.3f} (z = {z_perm:.2f})")

    out = {
        "n_rows": n, "n_unique": n_unique,
        "rows_by_compilation": {k: int(v) for k, v in
                                d.groupby("compilation").size().items()},
        "shared_pairs": {f"{a}x{b_}": len(p)
                         for (a, b_), p in pair_sets.items()},
        "residual_corr": {f"{a}x{b_}": rho[(a, b_)]
                          for (a, b_) in pair_sets},
        "joint_slope": [b, e], "A_joint": [b / SLOPE_TH,
                                           e / abs(SLOPE_TH)],
        "joint_dmu_exact_A": [bx, ex],
        "joint_two_component": {
            "A_gal": [bg / SLOPE_TH, eg / abs(SLOPE_TH)],
            "A_cl": [bc / SLOPE_TH, ec / abs(SLOPE_TH)]},
        "n_dedup": len(rep),
        "representative_sensitivity_slope_sd": float(np.std(alts)),
        "stacked_gls_consistency": [bgls, egls],
        "permutation": {"n_perm": N_PERM, "p_one_sided": p_perm,
                        "null_sigma": float(perm.std()),
                        "z_equiv": z_perm},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
