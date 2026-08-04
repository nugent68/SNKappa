#!/usr/bin/env python
"""Merge the deep-NIR noise decompositions into the fiducial noise CSVs.

The new fiducial kappa catalogs carry NirDirectFSF masses in the
VIDEO/UltraVISTA fields, so the latent-EIV noise tables must carry the
matching reduced classical M* variances there. This replaces each
survey's tracked latent_noise.csv rows with the NIR-run decompositions
(output/nir_video/latent_noise_nir.csv, computed with per-galaxy
sigma_nir) for the SNe in the NIR fields; all other rows keep the
original decomposition (identical estimator by construction).

Run: .venv/bin/python scripts/build_fiducial_noise.py
"""

import sys
from pathlib import Path

import pandas as pd

NIR = Path("output/nir_video/latent_noise_nir.csv")
TARGETS = {
    "des": Path("output/des_full/latent_noise.csv"),
    "union3": Path("output/union3/latent_noise.csv"),
    "pantheon": Path("output/pantheon/latent_noise.csv"),
}
COLS = ["rvar_gal_cl", "rvar_gal_tot", "rvar_cl_cl", "rvar_cl_tot",
        "var_zp"]


def main():
    nir = pd.read_csv(NIR)
    nir["CID"] = nir.CID.astype(str)
    for survey, path in TARGETS.items():
        old = pd.read_csv(path)
        old["CID"] = old.CID.astype(str)
        sub = nir[nir.survey.str.startswith(
            "des" if survey == "des" else survey)].drop_duplicates(
            "CID", keep="first").set_index("CID")
        hit = old.CID.isin(sub.index)
        for c in COLS:
            old.loc[hit, c] = sub.loc[old.CID[hit], c].to_numpy()
        bak = path.with_suffix(".prenir.csv")
        if not bak.exists():
            pd.read_csv(path).to_csv(bak, index=False)
        old.to_csv(path, index=False)
        print(f"{survey}: {int(hit.sum())}/{len(old)} rows replaced "
              f"with NIR decompositions -> {path}")


if __name__ == "__main__":
    main()
