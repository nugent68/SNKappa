#!/usr/bin/env python
"""Deep-NIR mass upgrade, Phase 2: build the NirDirect recalibration and
measure the scatter improvement against independent references.

Sample: DESI spec-z galaxies in the XMM-LSS (DES group X) and COSMOS
regional catalogs with deep-NIR photometry attached (snkappa.nir).

Recalibration (snkappa/data/nir_direct_fsf_recal.json): the established
FastSpecFit mass scale is transferred to the deep-NIR interpolation path
by fitting the sigmoid family of build_mstar_recal.py to
Delta = logM*(nir1um_fsf) - logM*(NirDirect raw) against (m, z), with m
on the corrected (FSF) scale. Outside the NIR footprints NirDirectFSF is
exactly nir1um_fsf, so the transfer only has to be right where NIR
photometry exists.

Independent scatter gate (output/nir_video/mstar_validation.json):
NMAD of (estimator - reference) for the OLD (nir1um_fsf) vs NEW
(nir_direct_fsf) estimators against
  (a) DESI DR1 CIGALE masses (desi_dr1.stellar_mass_emline, mass_cg) --
      both fields, the primary gate;
  (b) COSMOS2020 LePhare masses (lp_logmass) -- COSMOS bonus check.
Reference scatter is common to both comparisons, so the estimator
improvement is the QUADRATURE-removed scatter
sqrt(nmad_old^2 - nmad_new^2), not the raw NMAD difference (which is
diluted by the reference's own scatter). GATE: removed scatter vs
CIGALE >= 0.05 dex on the used_nir subset, else stop the sprint.

Run: .venv/bin/python scripts/nir_validate.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa.datalab import TapClient
from snkappa.nir import attach_nir
from snkappa.stellar import (Nir1umFSF, NirDirect, NirDirectFSF,
                             make_estimator)
from nir_fetch import des_foreground, cosmos_foreground

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


OUTDIR = Path("output/nir_video")
RECAL_PATH = Path("snkappa/data/nir_direct_fsf_recal.json")
Z_LO, Z_HI = 0.05, 1.10
GATE_QUAD_DEX = 0.05


def mags_dict(df):
    return {b: df[f"mag_{b}"].to_numpy(dtype=float)
            for b in ("g", "r", "i", "z", "y", "j", "h", "ks", "w1")
            if f"mag_{b}" in df}


def nmad(x):
    x = x[np.isfinite(x)]
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def build_sample():
    frames = []
    for name, loader in (("XMM", lambda: des_foreground("X")),
                         ("COSMOS", cosmos_foreground)):
        fg = loader()
        fg = attach_nir(fg)
        fg["field"] = name
        log(f"{name}: {len(fg)} galaxies, "
            f"{fg.attrs['n_nir_matched']} NIR-matched, "
            f"{int(fg.z_spec.notna().sum())} spec-z")
        frames.append(fg)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.z_spec.notna() & (df.z_spec > Z_LO)
            & (df.z_spec < Z_HI)].reset_index(drop=True)
    log(f"validation sample: {len(df)} spec-z galaxies")
    return df


def fit_recal(m_corr, z, delta, rng):
    """Sigmoid fit of build_mstar_recal.py, m on the corrected scale."""
    from scipy.optimize import least_squares
    Z_NODES = np.array([0.10, 0.30, 0.50, 0.70, 0.90, 1.05])

    def model(p, mm, zz):
        c0, m0, w = p[0], p[1], p[2]
        a = np.interp(zz, Z_NODES, p[3:])
        return c0 + a / (1.0 + np.exp(-(mm - m0) / np.clip(w, 0.05, 2.0)))

    sub = rng.choice(z.size, min(200000, z.size), replace=False)
    p0 = np.concatenate([[0.0, 10.5, 0.3], np.full(Z_NODES.size, 0.1)])
    # amplitude bound 0.8 (vs 0.6 in build_mstar_recal): the z ~ 0.1
    # massive-end cells genuinely need Delta ~ +0.4 (cell medians checked
    # against the fit 2026-08-03) and the 0.6 bound pinched the first node
    lo = np.concatenate([[-0.25, 9.5, 0.10], np.full(Z_NODES.size, -0.3)])
    hi = np.concatenate([[+0.25, 11.5, 1.00], np.full(Z_NODES.size, 0.8)])
    fit = least_squares(lambda p: model(p, m_corr[sub], z[sub]) - delta[sub],
                        p0, loss="soft_l1", f_scale=0.15, bounds=(lo, hi))
    p = fit.x
    resid = delta - model(p, m_corr, z)
    return {
        "description": ("Delta(m,z) = c0 + A(z) sigmoid((m-m0)/w), m on the "
                        "corrected (FSF) scale; apply by fixed point on the "
                        "raw NirDirect estimate"),
        "model": "sigmoid",
        "c0": float(p[0]), "m0": float(p[1]), "w": float(p[2]),
        "z_nodes": Z_NODES.tolist(), "a_nodes": p[3:].tolist(),
        "n_total": int(z.size),
        "resid_nmad_after": nmad(resid),
        "provenance": ("FSF scale transferred from nir1um_fsf on the "
                       "XMM-LSS + COSMOS DESI spec-z overlap with "
                       "VIDEO DR5 / COSMOS2020 photometry"),
    }


def fetch_cigale(df):
    """CIGALE logM* matched by position (<1 arcsec) to the sample."""
    tap = TapClient("https://datalab.noirlab.edu/tap", "cache")
    parts = []
    for ra0, dec0, rad in ((35.7, -4.99, 2.2), (150.1, 2.2, 2.2)):
        q = ("SELECT targetid, target_ra, target_dec, z, mass_cg "
             "FROM desi_dr1.stellar_mass_emline WHERE "
             f"'t'=Q3C_RADIAL_QUERY(target_ra, target_dec, {ra0}, {dec0}, "
             f"{rad}) AND mass_cg IS NOT NULL AND mass_cg > 0")
        parts.append(tap.query(q, label=f"cigale:{ra0}"))
    cg = pd.concat(parts, ignore_index=True)
    log(f"CIGALE reference: {len(cg)} rows")
    c_gal = SkyCoord(df.ra.to_numpy() * u.deg, df.dec.to_numpy() * u.deg)
    c_cg = SkyCoord(cg.target_ra.to_numpy() * u.deg,
                    cg.target_dec.to_numpy() * u.deg)
    idx, sep, _ = c_gal.match_to_catalog_sky(c_cg)
    good = (sep.arcsec < 1.0) \
        & (np.abs(cg.z.to_numpy()[idx] - df.z_spec.to_numpy()) < 0.01)
    ref = np.where(good, np.log10(cg.mass_cg.to_numpy()[idx]), np.nan)
    return ref


def fetch_lephare(df):
    """COSMOS2020 LePhare logM* matched by position (COSMOS only)."""
    nir = pd.read_parquet("data_nir/cosmos2020.parquet")
    nir = nir[np.isfinite(nir.lp_logmass) & (nir.lp_logmass > 6)]
    ref = np.full(len(df), np.nan)
    sel = df.field.to_numpy() == "COSMOS"
    if not sel.any():
        return ref
    c_gal = SkyCoord(df.ra.to_numpy()[sel] * u.deg,
                     df.dec.to_numpy()[sel] * u.deg)
    c_nir = SkyCoord(nir.ra.to_numpy() * u.deg, nir.dec.to_numpy() * u.deg)
    idx, sep, _ = c_gal.match_to_catalog_sky(c_nir)
    good = sep.arcsec < 1.0
    vals = np.where(good, nir.lp_logmass.to_numpy()[idx], np.nan)
    ref[sel] = vals
    return ref


def compare(tag, est_old, est_new, df, ref, out):
    """NMAD of old/new estimator against a reference, overall + binned."""
    z = df.z_spec.to_numpy(dtype=float)
    mags = mags_dict(df)
    old = est_old.logmstar(mags, z)
    new, used = est_new.logmstar_flagged(mags, z)
    ok = np.isfinite(ref) & used          # NIR-interpolated galaxies only
    d_old = (old - ref)[ok]
    d_new = (new - ref)[ok]
    block = {"n": int(ok.sum()),
             "nmad_old": nmad(d_old), "nmad_new": nmad(d_new),
             "med_old": float(np.median(d_old)),
             "med_new": float(np.median(d_new))}
    zbins = ((0.05, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.1))
    block["z_bins"] = {}
    zz = z[ok]
    for zlo, zhi in zbins:
        s = (zz >= zlo) & (zz < zhi)
        if s.sum() < 100:
            continue
        block["z_bins"][f"{zlo}-{zhi}"] = {
            "n": int(s.sum()), "nmad_old": nmad(d_old[s]),
            "nmad_new": nmad(d_new[s])}
    # massive end, binned on the REFERENCE (selection-noise safe)
    hi = ref[ok] > 11.0
    if hi.sum() > 50:
        block["massive_refgt11"] = {
            "n": int(hi.sum()), "nmad_old": nmad(d_old[hi]),
            "nmad_new": nmad(d_new[hi]),
            "med_old": float(np.median(d_old[hi])),
            "med_new": float(np.median(d_new[hi]))}
    out[tag] = block
    log(f"{tag}: N={block['n']} NMAD old {block['nmad_old']:.3f} -> "
        f"new {block['nmad_new']:.3f} dex")


def main():
    rng = np.random.default_rng(20130901)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.352)

    df = build_sample()
    z = df.z_spec.to_numpy(dtype=float)
    mags = mags_dict(df)

    old_fsf = Nir1umFSF(cosmo)
    m_old_corr = old_fsf.logmstar(mags, z)
    new_raw, used = NirDirect(cosmo).logmstar_flagged(mags, z)
    frac_nir = used.mean()
    log(f"used_nir fraction of spec-z sample: {frac_nir:.3f} "
        f"({int(used.sum())} galaxies)")

    # ---- recal fit (NIR-interpolated galaxies only) ----------------------
    sel = used & np.isfinite(m_old_corr) & np.isfinite(new_raw)
    delta = (m_old_corr - new_raw)[sel]
    tab = fit_recal(m_old_corr[sel], z[sel], delta, rng)
    RECAL_PATH.write_text(json.dumps(tab, indent=1))
    log(f"recal table saved ({tab['n_total']} galaxies, "
        f"c0={tab['c0']:+.3f}, A(z)={np.round(tab['a_nodes'],3).tolist()}, "
        f"resid nmad {tab['resid_nmad_after']:.3f})")

    new_fsf = NirDirectFSF(cosmo)     # reload with the fresh table
    assert new_fsf.has_nir_table

    # ---- independent references -----------------------------------------
    out = {"n_sample": int(len(df)), "frac_used_nir": float(frac_nir),
           "recal": {k: tab[k] for k in
                     ("c0", "m0", "w", "z_nodes", "a_nodes", "n_total",
                      "resid_nmad_after")}}
    ref_cg = fetch_cigale(df)
    compare("vs_cigale", old_fsf, new_fsf, df, ref_cg, out)
    ref_lp = fetch_lephare(df)
    compare("vs_lephare_cosmos", old_fsf, new_fsf, df, ref_lp, out)

    g = out["vs_cigale"]
    impr = g["nmad_old"] - g["nmad_new"]
    quad = float(np.sqrt(max(g["nmad_old"] ** 2 - g["nmad_new"] ** 2, 0.0)))
    out["gate"] = {"threshold_removed_quadrature_dex": GATE_QUAD_DEX,
                   "nmad_improvement_dex": round(impr, 4),
                   "removed_scatter_quadrature_dex": round(quad, 4),
                   "pass": quad >= GATE_QUAD_DEX}
    (OUTDIR / "mstar_validation.json").write_text(
        json.dumps(out, indent=2, default=float))
    log(f"saved {OUTDIR/'mstar_validation.json'}")
    log(f"GATE {'PASS' if out['gate']['pass'] else 'FAIL'} "
        f"(NMAD improvement {impr:+.3f} dex, "
        f"removed scatter {quad:.3f} dex in quadrature)")


if __name__ == "__main__":
    main()
