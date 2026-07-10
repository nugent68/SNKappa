#!/usr/bin/env python
"""Extract DES-like mock regions from cosmoDC2 v1.1.4 (TODO 5.1).

Runs on Perlmutter (NERSC). Each cosmoDC2 healpix pixel (nside=32,
~3.4 deg^2) becomes one independent mock region, playing the role of a DES
field group. Per pixel, writes three compact parquet files to --outdir:

- fg_<pix>.parquet   foreground catalog: dereddened-depth-matched
                     (mag_z_lsst <= 22.5, the DES analysis depth) galaxies
                     with TRUE redshift and stellar mass (observational
                     noise is forward-modeled later, in mock_calibration.py)
- src_<pix>.parquet  mock supernova sightlines: galaxies at 0.15 < z < 1.13
                     carrying the ray-traced TRUE convergence at their
                     position and redshift
- cl_<pix>.parquet   mock cluster catalog: central galaxies of halos with
                     M_halo >= 6e13 Msun (~ the Wen & Han completeness),
                     with TRUE halo mass (richness scatter added later)

Row groups in these files are redshift slices (0-1, 1-2, 2-3); only the
first two are read.

Run: ./venv/bin/python scripts/mock_extract_dc2.py --pixels 10066 10195 ...
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DC2 = Path("/global/cfs/cdirs/lsst/shared/xgal/cosmoDC2/"
           "cosmoDC2_v1.1.4_parquet")
COLS = ["ra", "dec", "redshift_true", "stellar_mass", "halo_mass",
        "is_central", "convergence", "mag_z_lsst", "mag_i_lsst"]
MAG_LIMIT = 22.5
N_SRC = 450


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", type=int, nargs="+", required=True)
    ap.add_argument("--outdir", default="mockdata")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20130901)

    for pix in args.pixels:
        f = pq.ParquetFile(DC2 / f"cosmoDC2_v1.1.4_image_healpix{pix}.parquet")
        parts = [f.read_row_group(g, columns=COLS).to_pandas()
                 for g in range(min(2, f.metadata.num_row_groups))]
        d = pd.concat(parts, ignore_index=True)
        d = d[d.redshift_true < 1.45]

        fg = d[(d.mag_z_lsst <= MAG_LIMIT) & (d.stellar_mass > 1e7)][
            ["ra", "dec", "redshift_true", "stellar_mass", "mag_z_lsst"]]
        fg.to_parquet(outdir / f"fg_{pix}.parquet", index=False)

        # mock SN sightlines: host-like galaxies, stratified uniformly in z
        s = d[(d.redshift_true > 0.15) & (d.redshift_true < 1.13)
              & (d.mag_i_lsst < 24.5)]
        zq = np.linspace(0.15, 1.13, 25)
        keep = []
        n_per = int(np.ceil(N_SRC / (zq.size - 1)))
        for lo, hi in zip(zq[:-1], zq[1:]):
            idx = s.index[(s.redshift_true >= lo) & (s.redshift_true < hi)]
            if idx.size:
                keep.append(rng.choice(idx, min(n_per, idx.size),
                                       replace=False))
        src = s.loc[np.concatenate(keep)][
            ["ra", "dec", "redshift_true", "convergence", "mag_i_lsst"]]
        src.to_parquet(outdir / f"src_{pix}.parquet", index=False)

        cl = d[(d.is_central) & (d.halo_mass >= 6e13)
               & (d.redshift_true < 1.2)][
            ["ra", "dec", "redshift_true", "halo_mass"]]
        cl.to_parquet(outdir / f"cl_{pix}.parquet", index=False)
        print(f"pixel {pix}: fg {len(fg)}, src {len(src)}, cl {len(cl)}",
              flush=True)


if __name__ == "__main__":
    main()
