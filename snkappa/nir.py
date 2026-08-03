"""Attach deep-NIR photometry (VIDEO DR5 / COSMOS2020) to a regional catalog.

Left-joins Y/J/H/Ks AUTO magnitudes (AB -- verified empirically against LS
z-band in scripts/nir_fetch.py's audit) onto the clean_and_merge output by
nearest-neighbor position match. Galaxies without a counterpart (outside
the VIDEO/UltraVISTA footprints, or in mask holes) keep NaN and the
NirDirect estimator reduces to the z <-> W1 path for them.

Magnitudes are dereddened with E(B-V) derived from the galaxy's own LS
mw_transmission_z (A_z = 1.211 E(B-V) for DECam z), using VISTA
Fitzpatrick-99 coefficients; the correction is < 0.03 mag in all four
fields (E(B-V) <= 0.03) so coefficient choice is immaterial.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

NIR_BANDS = ("y", "j", "h", "ks")

# A_lambda / E(B-V): DECam z (LS convention) and VISTA YJHKs (F99)
R_LS_Z = 1.211
R_NIR = {"y": 1.017, "j": 0.705, "h": 0.441, "ks": 0.308}

NIR_FILES = ("video_xmm.parquet", "video_cdfs.parquet",
             "video_es1.parquet", "cosmos2020.parquet")


def _load_overlapping(df, data_dir: Path) -> pd.DataFrame | None:
    """Concatenate NIR catalogs whose bounding box overlaps the region."""
    ra0, ra1 = df["ra"].min(), df["ra"].max()
    d0, d1 = df["dec"].min(), df["dec"].max()
    parts = []
    for f in NIR_FILES:
        p = data_dir / f
        if not p.exists():
            continue
        nir = pd.read_parquet(p)
        if (nir["ra"].min() > ra1 or nir["ra"].max() < ra0
                or nir["dec"].min() > d1 or nir["dec"].max() < d0):
            continue
        # VIDEO scattered-light halo regions: photometry unreliable
        if "haloflag" in nir and nir["haloflag"].notna().any():
            nir = nir[~(nir["haloflag"].fillna(0) > 0)]
        parts.append(nir)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def attach_nir(df: pd.DataFrame, data_dir="data_nir",
               match_arcsec: float = 1.0) -> pd.DataFrame:
    """Add mag_y/j/h/ks (+ magerr_*) columns; NaN where no counterpart."""
    df = df.copy()
    for b in NIR_BANDS:
        df[f"mag_{b}"] = np.nan
        df[f"magerr_{b}"] = np.nan
    nir = _load_overlapping(df, Path(data_dir))
    df.attrs["n_nir_matched"] = 0
    if nir is None or not len(df):
        return df

    c_gal = SkyCoord(df["ra"].to_numpy() * u.deg,
                     df["dec"].to_numpy() * u.deg)
    c_nir = SkyCoord(nir["ra"].to_numpy() * u.deg,
                     nir["dec"].to_numpy() * u.deg)
    idx, sep, _ = c_gal.match_to_catalog_sky(c_nir)
    good = sep.arcsec < match_arcsec

    # per-galaxy E(B-V) from the LS z-band MW transmission
    mwt = df["mw_transmission_z"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_z = -2.5 * np.log10(np.clip(mwt, 1e-6, 1.0))
    ebv = np.where((mwt > 0) & (mwt <= 1.0), a_z / R_LS_Z, 0.0)

    for b in NIR_BANDS:
        m = nir[f"{b}_mag_auto"].to_numpy(dtype=float)[idx]
        e = nir[f"{b}_magerr_auto"].to_numpy(dtype=float)[idx]
        m = m - R_NIR[b] * ebv
        df.loc[good, f"mag_{b}"] = m[good]
        df.loc[good, f"magerr_{b}"] = e[good]
    df.attrs["n_nir_matched"] = int(good.sum())
    return df
