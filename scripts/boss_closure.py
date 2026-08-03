#!/usr/bin/env python
"""Public DeltaSigma closure: SNKappa halo-chain prediction vs our own
BOSS x KiDS-Legacy measurement (Tier-2 replacement for the DESI-internal
Lensing Without Borders comparison; everything public and reproducible).

Prediction (identical chain to the SN analysis and the LWB-era closure):
LS DR10 photometry (via the public ls_dr10 x sdss_dr17.specobj
crossmatch) -> nir1um_fsf stellar mass at the BOSS spec-z -> HMF-weighted
posterior <Mh|M*> -> cap 10^13.8 -> Diemer & Joyce c(M,z) -> BMO
DeltaSigma + stellar point mass; stacked over the IDENTICAL per-bin lens
tables used in the measurement (scripts/boss_kids_measure.py), Planck15,
comoving conventions matched.

    A_DS = sum w ds_meas ds_pred / sum w ds_pred^2  per window

Run:  .venv/bin/python scripts/boss_closure.py [--mstar-method ...]
Outputs: output/delta_sigma/boss_closure{_variant}.json (fully public).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.config import HaloModelConfig
from snkappa.datalab import TapClient
from snkappa.halos import HaloModel, bmo_table
from snkappa.kappa import angular_sep_arcsec
from snkappa.stellar import make_estimator

DATA = Path("data_boss_kids")
MEAS = Path("output/delta_sigma/boss_kids_meas.json")
RA_CHUNKS = np.arange(110.0, 271.0, 20.0)   # KiDS-N strip coverage
DEC_RANGE = (-6.0, 6.0)

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def fetch_ls_photometry():
    """LS DR10 fluxes for SDSS specobj in the KiDS-N strip (cached)."""
    tap = TapClient("https://datalab.noirlab.edu/tap", "cache")
    frames = []
    for lo, hi in zip(RA_CHUNKS[:-1], RA_CHUNKS[1:]):
        q = ("SELECT x.ra2, x.dec2, x.distance, "
             "t.flux_g, t.flux_r, t.flux_z, t.flux_w1, "
             "t.mw_transmission_g, t.mw_transmission_r, "
             "t.mw_transmission_z, t.mw_transmission_w1, t.flux_ivar_w1 "
             "FROM ls_dr10.x1p5__tractor__sdss_dr17__specobj AS x "
             "JOIN ls_dr10.tractor AS t ON t.ls_id = x.id1 "
             f"WHERE x.dec2 BETWEEN {DEC_RANGE[0]} AND {DEC_RANGE[1]} "
             f"AND x.ra2 BETWEEN {lo} AND {hi}")
        frames.append(tap.query(q, label=f"xmatch:{lo:.0f}"))
        log(f"  xmatch chunk RA {lo:.0f}-{hi:.0f}: {len(frames[-1])} rows")
    xm = pd.concat(frames, ignore_index=True)
    # keep the closest LS counterpart per SDSS position
    xm = xm.sort_values("distance").drop_duplicates(
        ["ra2", "dec2"], keep="first").reset_index(drop=True)
    return xm


def dered(xm, b):
    f = xm[f"flux_{b}"].to_numpy(dtype=float)
    mw = xm[f"mw_transmission_{b}"].to_numpy(dtype=float)
    ok = (f > 0) & (mw > 0) & (mw <= 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = 22.5 - 2.5 * np.log10(np.where(ok, f / np.where(ok, mw, 1),
                                           np.nan))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mstar-method", default="nir1um_fsf")
    ap.add_argument("--smhm-inverse", default="posterior",
                    choices=("posterior", "naive"))
    ap.add_argument("--variant", default="")
    args = ap.parse_args()

    from astropy.cosmology import Planck15 as cosmo
    meas = json.loads(MEAS.read_text())
    # dsigma little-h conventions (verified in dsigma/precompute.py):
    # bare rp bins are comoving Mpc/h; DeltaSigma returns h Msun/pc^2.
    # Convert to h-free comoving Mpc and Msun/pc^2 to match the
    # prediction side.
    H_LITTLE = cosmo.H0.value / 100.0
    xm = fetch_ls_photometry()
    mags_all = {b: dered(xm, b) for b in ("g", "r", "z", "w1")}
    w1_snr = (xm.flux_w1.to_numpy(float)
              * np.sqrt(np.clip(xm.flux_ivar_w1.to_numpy(float), 0, None)))
    mags_all["w1"] = np.where(w1_snr > 2.0, mags_all["w1"], np.nan)
    log(f"LS photometry for {len(xm)} specobj positions")

    hcfg = HaloModelConfig(smhm_inverse=args.smhm_inverse)
    hm = HaloModel(hcfg, cosmo, 1.2)
    est = make_estimator(args.mstar_method, cosmo)
    tab = bmo_table()

    out = {"args": vars(args),
           "provenance": ("prediction: LS DR10 photometry via public "
                          "ls_dr10 x sdss_dr17 specobj crossmatch, "
                          "SNKappa halo chain; measurement: "
                          "boss_kids_meas.json (BOSS x KiDS-Legacy, "
                          "public)"), "bins": []}
    for i, mb in enumerate(meas["bins"]):
        lens = pd.read_parquet(DATA / f"lens_bin{i}.parquet")
        # local nearest-position match lens -> specobj xmatch (< 1")
        li, xi = [], []
        xra, xdec = xm.ra2.to_numpy(), xm.dec2.to_numpy()
        # coarse dec presort for speed
        order = np.argsort(xdec)
        xdec_s = xdec[order]
        for j, (ra0, dec0) in enumerate(zip(lens.ra, lens.dec)):
            k0 = np.searchsorted(xdec_s, dec0 - 0.001)
            k1 = np.searchsorted(xdec_s, dec0 + 0.001)
            if k1 <= k0:
                continue
            cand = order[k0:k1]
            sep = angular_sep_arcsec(ra0, dec0, xra[cand], xdec[cand])
            k = np.argmin(sep)
            if sep[k] < 1.0:
                li.append(j); xi.append(cand[k])
        li, xi = np.array(li), np.array(xi)
        z = lens.z.to_numpy()[li]
        w = lens.w_sys.to_numpy()[li]
        mags = {b: v[xi] for b, v in mags_all.items()}
        logms = est.logmstar(mags, z)
        ok = np.isfinite(logms)
        z, w, logms = z[ok], w[ok], logms[ok]
        log(f"bin {i} ({mb['sample']} z[{mb['zmin']},{mb['zmax']}]): "
            f"{len(lens)} lenses -> {ok.sum()} matched+usable")

        ib = hm.zbin_index(z)
        rhos, rs, tau = hm.halo_params(logms, ib)
        rp = np.array(mb["rp"]) / H_LITTLE          # -> comoving Mpc
        r_phys = rp[None, :] / (1.0 + z[:, None])
        x = r_phys / rs[:, None]
        ds_h = (rhos * rs)[:, None] * tab.delta_sigma_dimless(
            x, np.broadcast_to(tau[:, None], x.shape))
        ds_star = 10.0 ** logms[:, None] / (np.pi * (r_phys * 1e6) ** 2)
        ds_com = (ds_h / 1e12 + ds_star) / (1.0 + z[:, None]) ** 2
        pred = np.average(ds_com, axis=0, weights=w)

        r200 = rs * tau
        r200_med = float(np.median(r200))
        r1h = 2.0 * r200_med * (1.0 + float(np.median(z)))
        windows = {"one_halo": (0.08, max(0.25, r1h)),
                   "fiducial": (0.10, 1.00)}
        dsm = np.array(mb["ds"]) * H_LITTLE         # -> Msun/pc^2
        dse = np.array(mb["ds_err"]) * H_LITTLE
        entry = {"sample": mb["sample"], "zmin": mb["zmin"],
                 "zmax": mb["zmax"], "n_used": int(z.size),
                 "logms_med": float(np.median(logms)),
                 "r200_med_phys": r200_med,
                 "rp": rp.tolist(), "ds_pred": pred.tolist(),
                 "ds_meas": dsm.tolist(), "ds_err": dse.tolist(),
                 "windows": {}}
        for wname, (rlo, rhi) in windows.items():
            m = (rp > rlo) & (rp < rhi) & (dse > 0)
            if m.sum() < 2:
                continue
            wt = 1.0 / dse[m] ** 2
            a = np.sum(wt * dsm[m] * pred[m]) / np.sum(wt * pred[m] ** 2)
            ea = 1.0 / np.sqrt(np.sum(wt * pred[m] ** 2))
            entry["windows"][wname] = {"rp_range": [rlo, rhi],
                                       "A_ds": float(a),
                                       "A_ds_err": float(ea)}
        oh = entry["windows"].get("one_halo", {})
        log(f"  A_1h = {oh.get('A_ds', float('nan')):.3f} "
            f"+- {oh.get('A_ds_err', float('nan')):.3f} | "
            f"logM*_med = {entry['logms_med']:.2f}")
        out["bins"].append(entry)

    suffix = f"_{args.variant}" if args.variant else ""
    dst = Path(f"output/delta_sigma/boss_closure{suffix}.json")
    dst.write_text(json.dumps(out, indent=1, default=float))
    log(f"saved {dst}")


if __name__ == "__main__":
    main()
