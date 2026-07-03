"""Hybrid stellar masses: FrankenBlast SBI++ for the galaxies that matter,
calibrated cheap estimator for the rest.

Export side:  fb_targets.csv = G1/G2-style must-fits + the top-n_top kappa_i
contributors along the SN sightline + an n_calib random calibration sample
(spec-z-weighted) from the regional catalog. Fluxes are exported already
MW-dereddened (Legacy Surveys mw_transmission) in maggies, mapped onto the
FrankenBlast trained filter set (DES_g/r/z for LS grz -- BASS/MzLS in the
north differs from DECam by a few percent, absorbed by the SBI noise model --
and WISE_W1/W2).

Ingest side: per-galaxy surviving log M* posteriors replace the cheap
estimate where fitted; the median offset (cheap - FB) measured on ALL fitted
galaxies is subtracted from the cheap estimator everywhere else (including
every random sightline, keeping the mass-sheet zero point consistent), and
the robust scatter replaces the assumed mstar_scatter_dex in the Monte Carlo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# snkappa band -> FrankenBlast filter name
BAND_MAP = {"g": "DES_g", "r": "DES_r", "z": "DES_z",
            "w1": "WISE_W1", "w2": "WISE_W2"}


def _maggies(df, band):
    """Dereddened maggies and uncertainties (NaN-safe)."""
    flux = df[f"flux_{band}"].to_numpy(dtype=float)
    ivar = df[f"flux_ivar_{band}"].to_numpy(dtype=float)
    mw = df[f"mw_transmission_{band}"].to_numpy(dtype=float)
    bad = (flux <= -9000) | (mw <= 0) | (mw > 1) | (ivar <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 1e-9 * flux / mw
        e = 1e-9 / (np.sqrt(ivar) * mw)
    m[bad] = np.nan
    e[bad] = np.nan
    return m, e


def export_targets(cfg, df, sn_idx, sn_kappa, outdir: Path, rng) -> Path:
    """Write fb_targets.csv; returns its path."""
    fb = cfg.frankenblast
    z_best = df["z_spec"].to_numpy(dtype=float).copy()
    z_src_flag = np.where(np.isfinite(z_best), "spec", "phot")
    z_best = np.where(np.isfinite(z_best),
                      z_best, df["zp_med"].to_numpy(dtype=float))

    # top kappa contributors along the SN sightline
    order = np.argsort(sn_kappa)[::-1]
    top = set(np.asarray(sn_idx)[order[:fb.n_top]].tolist())

    # calibration sample: spec-z weighted 4:1 over the regional catalog
    is_spec = df["z_spec"].notna().to_numpy()
    pool_spec = np.flatnonzero(is_spec)
    pool_phot = np.flatnonzero(~is_spec)
    n_spec = min(int(0.8 * fb.n_calib), pool_spec.size)
    n_phot = min(fb.n_calib - n_spec, pool_phot.size)
    calib = set(rng.choice(pool_spec, n_spec, replace=False).tolist())
    calib |= set(rng.choice(pool_phot, n_phot, replace=False).tolist())

    # deflector-region galaxies (G1/G2): excluded from kappa_ext but fitted
    # as validation anchors against published SED masses
    defl = set(np.flatnonzero(df["excl_deflector"].to_numpy()).tolist())

    sel = sorted(top | calib | defl)
    sub = df.iloc[sel]
    z_defl = cfg.deflector.z_lens

    def role(i):
        if i in defl:
            return "deflector"
        return "top" if i in top else "calib"

    out = pd.DataFrame({
        "ls_id": sub["ls_id"].astype(np.int64).to_numpy(),
        "z": [z_defl if i in defl else z_best[i] for i in sel],
        "z_source": ["lens" if i in defl else z_src_flag[i] for i in sel],
        "role": [role(i) for i in sel],
    })
    for band, fname in BAND_MAP.items():
        if f"flux_{band}" not in df.columns:
            continue
        m, e = _maggies(sub, band)
        out[f"maggies_{fname}"] = m
        out[f"maggies_unc_{fname}"] = e
    out = out[np.isfinite(out["z"]) & (out["z"] > 0)]
    path = outdir / "fb_targets.csv"
    out.to_csv(path, index=False)
    return path


def load_and_calibrate(cfg, df, stellar_est):
    """Read fb_results.csv; return (logms_override, logms_err, calib dict).

    logms_override/logms_err are arrays over df rows (NaN where not fitted).
    calib holds the offset/scatter of (cheap - FB) measured on the fitted
    sample; the offset is later SUBTRACTED from the cheap estimator.
    """
    path = Path(cfg.frankenblast.results_path)
    res = pd.read_csv(path)
    res = res[np.isfinite(res["logms_fb"])].drop_duplicates(
        "ls_id", keep="last")

    n = len(df)
    override = np.full(n, np.nan)
    err = np.full(n, np.nan)
    pos = pd.Series(np.arange(n), index=df["ls_id"].astype(np.int64))
    matched = res[res["ls_id"].isin(pos.index)]
    idx = pos.loc[matched["ls_id"].astype(np.int64)].to_numpy()
    override[idx] = matched["logms_fb"].to_numpy()
    err[idx] = np.clip(matched["logms_fb_err"].to_numpy(), 0.05, 1.0)

    # calibration: cheap estimator evaluated at the SAME redshift used by FB
    mags = {b: df[f"mag_{b}"].to_numpy(dtype=float)[idx]
            for b in ("g", "r", "i", "z", "w1") if f"mag_{b}" in df}
    cheap = stellar_est.logmstar(mags, matched["z"].to_numpy(dtype=float))
    diff = cheap - matched["logms_fb"].to_numpy()
    good = np.isfinite(diff)
    p16, p50, p84 = np.percentile(diff[good], [16, 50, 84])
    calib = {
        "n_fitted": int(good.sum()),
        "offset_cheap_minus_fb_dex": float(p50),
        "scatter_dex": float(0.5 * (p84 - p16)),
        "results_path": str(path),
    }
    return override, err, calib
