#!/usr/bin/env python
"""Measure DeltaSigma for BOSS DR12 lenses with KiDS-Legacy sources.

Fully public inputs (SDSS DR12 LSS catalogs; KiDS-Legacy DR5 shear +
KiDZ calibration), measured with dsigma (Lange & Huang) following the
package's documented KiDS-Legacy recipe: Planck15, comoving separations,
scalar shear response correction, random subtraction, no boost, 100-field
jackknife errors. This is the public replacement for the DESI-internal
Lensing Without Borders comparison (validation (v)): anyone can rerun it.

Lens bins (the dsigma/Amon-style BOSS bins):
    LOWZ  0.15-0.31, 0.31-0.43;  CMASS 0.43-0.54, 0.54-0.70.
Footprint: lenses/randoms restricted to 0.5-deg cells occupied by KiDS
sources; the identical per-bin lens tables are saved for the prediction
side (closure compares prediction and measurement over the SAME lenses).

Run (after downloads): .venv/bin/python scripts/boss_kids_measure.py
Outputs: output/delta_sigma/boss_kids_meas.json,
         data_boss_kids/lens_bin*.parquet
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path("data_boss_kids").resolve()
OUT = Path("output/delta_sigma/boss_kids_meas.json").resolve()
Z_BINS = [(0.15, 0.31, "LOWZ"), (0.31, 0.43, "LOWZ"),
          (0.43, 0.54, "CMASS"), (0.54, 0.70, "CMASS")]
RP_BINS = np.logspace(np.log10(0.08), np.log10(3.0), 11)
RAND_FACTOR = 8

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def load_boss(kind):
    """(ra, dec, z, w) concatenated LOWZ+CMASS North tables."""
    rows = []
    for samp in ("LOWZ", "CMASS"):
        t = Table.read(DATA / f"{kind}_DR12v5_{samp}_North.fits.gz")
        d = pd.DataFrame({
            "ra": np.asarray(t["RA"], float),
            "dec": np.asarray(t["DEC"], float),
            "z": np.asarray(t["Z"], float),
            "sample": samp})
        if kind == "galaxy":
            d["w_sys"] = (np.asarray(t["WEIGHT_SYSTOT"], float)
                          * (np.asarray(t["WEIGHT_CP"], float)
                             + np.asarray(t["WEIGHT_NOZ"], float) - 1.0))
        else:
            d["w_sys"] = 1.0
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    # LSS catalogs carry occasional sentinel redshifts
    return out[(out.z > 0.01) & (out.z < 1.0)].reset_index(drop=True)


def main():
    os.chdir(DATA)
    if not Path("kids_legacy.hdf5").exists():
        log("processing KiDS-Legacy raw catalogs (one-time)...")
        from dsigma.scripts.process_kids_legacy import process_kids_legacy
        process_kids_legacy()
    table_s = Table.read("kids_legacy.hdf5", path="catalog")
    table_n = Table.read("kids_legacy.hdf5", path="calibration")
    table_s["z_l_max"] = np.array(
        [0.1, 0.42, 0.58, 0.71, 0.90, 1.14])[table_s["z_bin"]] - 0.1
    log(f"KiDS sources: {len(table_s)}")

    # occupancy footprint (0.5 deg cells) from a source subsample
    sub = table_s[::47]
    cells = set(zip((np.asarray(sub["ra"]) * 2).astype(int),
                    ((np.asarray(sub["dec"]) + 90) * 2).astype(int)))

    def in_kids(d):
        key = list(zip((d.ra.to_numpy() * 2).astype(int),
                       ((d.dec.to_numpy() + 90) * 2).astype(int)))
        return np.array([k in cells for k in key])

    lens = load_boss("galaxy")
    rand = load_boss("random0")
    lens = lens[in_kids(lens)].reset_index(drop=True)
    rand = rand[in_kids(rand)].reset_index(drop=True)
    rng = np.random.default_rng(20130901)
    if len(rand) > RAND_FACTOR * len(lens):
        rand = rand.iloc[rng.choice(len(rand), RAND_FACTOR * len(lens),
                                    replace=False)].reset_index(drop=True)
    log(f"BOSS in KiDS footprint: {len(lens)} lenses, {len(rand)} randoms")

    from dsigma.precompute import precompute
    from dsigma.stacking import excess_surface_density
    from dsigma.jackknife import (compute_jackknife_fields,
                                  jackknife_resampling)
    from astropy.cosmology import Planck15

    table_l = Table.from_pandas(lens[["ra", "dec", "z", "w_sys"]])
    table_r = Table.from_pandas(rand[["ra", "dec", "z", "w_sys"]])
    kwargs = dict(cosmology=Planck15, comoving=True, table_n=table_n,
                  n_jobs=max(os.cpu_count() - 2, 1))
    log("precompute lenses...")
    precompute(table_l, table_s, RP_BINS, **kwargs)
    log("precompute randoms...")
    precompute(table_r, table_s, RP_BINS, **kwargs)

    keep_l = np.sum(table_l["sum 1"], axis=1) > 0
    keep_r = np.sum(table_r["sum 1"], axis=1) > 0
    table_l, lens = table_l[keep_l], lens[np.asarray(keep_l)]
    table_r = table_r[keep_r]
    centers = compute_jackknife_fields(
        table_l, 100, weights=np.sum(table_l["sum 1"], axis=1))
    compute_jackknife_fields(table_r, centers)

    skw = dict(scalar_shear_response_correction=True,
               random_subtraction=True)
    rp_mid = np.sqrt(RP_BINS[1:] * RP_BINS[:-1])
    results = {"provenance": ("BOSS DR12 LSS (public SAS) x KiDS-Legacy "
                              "DR5 (public); dsigma 1.2.2, documented "
                              "KiDS recipe: Planck15, comoving, scalar "
                              "shear response, random subtraction, "
                              "100-field jackknife"),
               "rp_bins": RP_BINS.tolist(), "bins": []}
    for i, (zlo, zhi, samp) in enumerate(Z_BINS):
        ml = (zlo <= np.asarray(table_l["z"])) \
            & (np.asarray(table_l["z"]) < zhi)
        mr = (zlo <= np.asarray(table_r["z"])) \
            & (np.asarray(table_r["z"]) < zhi)
        tl, tr = table_l[ml], table_r[mr]
        ds = excess_surface_density(tl, table_r=tr, **skw)
        err = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, tl, table_r=tr, **skw)))
        results["bins"].append({
            "sample": samp, "zmin": zlo, "zmax": zhi,
            "n_lens": int(ml.sum()),
            "rp": rp_mid.tolist(), "ds": np.asarray(ds, float).tolist(),
            "ds_err": np.asarray(err, float).tolist()})
        log(f"{samp} z[{zlo},{zhi}]: N={int(ml.sum())} "
            f"ds(0.1-0.3Mpc)~{np.asarray(ds)[1]:.1f}+-{err[1]:.1f}")
        lens[ml].reset_index(drop=True).to_parquet(
            DATA / f"lens_bin{i}.parquet")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, default=float))
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
