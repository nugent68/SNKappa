"""Catalog hygiene: Data Lab NULL sentinels must never fabricate photometry,
band requirements must not drop usable galaxies, and the DESI crossmatch
must resolve fiber collisions by distance."""

import numpy as np
import pandas as pd

from snkappa.catalog import clean_and_merge, dered_mag


def _tractor_frame():
    """Three galaxies: A = g-undetected red galaxy WITH W1 (must survive);
    B = optical-only, no W1 (must survive via the Taylor fallback);
    C = neither g nor W1 (unusable, must be dropped)."""
    n = 3
    f_z = 10.0 ** ((22.5 - 19.0) / 2.5)   # mag_z = 19, well inside the cut
    df = pd.DataFrame({
        "ra": [150.05, 150.10, 150.15],
        "dec": [30.0, 30.0, 30.0],
        "maskbits": [0, 0, 0],
        "flux_g": [-9999.0, f_z * 2.0, -9999.0],
        "flux_r": [f_z, f_z, f_z],
        "flux_i": [f_z, f_z, f_z],
        "flux_z": [f_z, f_z, f_z],
        "flux_w1": [10.0, 10.0, -9999.0],
        "flux_w2": [-9999.0] * n,
        "flux_ivar_w1": [1.0, 0.01, 1.0],   # B: SNR = 1 < 2 -> W1 masked
        "z_phot_median": [0.4] * n,
        "z_phot_std": [0.08] * n,
        "z_phot_l68": [0.32] * n,
        "z_phot_u68": [0.48] * n,
    })
    for b in ("g", "r", "i", "z", "w1", "w2"):
        df[f"mw_transmission_{b}"] = 1.0
    for b in ("r", "z"):
        df[f"fracflux_{b}"] = 0.1
        df[f"fracmasked_{b}"] = 0.1
        df[f"fracin_{b}"] = 0.9
    return df


def _zpix_frame(rows):
    return pd.DataFrame(
        [(ra, dec, z, "t", 0, 100.0) for ra, dec, z in rows],
        columns=["mean_fiber_ra", "mean_fiber_dec", "z",
                 "zcat_primary", "coadd_fiberstatus", "deltachi2"])


def test_band_requirement_keeps_w1_only_red_galaxy(cfg):
    out = clean_and_merge(cfg, _tractor_frame(), _zpix_frame([]))
    # A (W1, no g) and B (g, no W1) survive; C (neither) is dropped
    assert sorted(out["ra"].tolist()) == [150.05, 150.10]
    a = out[out.ra == 150.05].iloc[0]
    assert np.isnan(a["mag_g"]) and np.isfinite(a["mag_w1"])
    b = out[out.ra == 150.10].iloc[0]
    assert np.isfinite(b["mag_g"]) and np.isnan(b["mag_w1"])


def test_specz_crossmatch_keeps_closest_fiber(cfg):
    # two valid fibers within 1 arcsec of galaxy A; the FARTHER one is
    # listed last, so last-write-wins would pick z = 0.9
    zpix = _zpix_frame([
        (150.05, 30.0 + 0.3 / 3600.0, 0.5),   # 0.3 arcsec, closest
        (150.05, 30.0 + 0.8 / 3600.0, 0.9),   # 0.8 arcsec
    ])
    out = clean_and_merge(cfg, _tractor_frame(), zpix)
    a = out[out.ra == 150.05].iloc[0]
    assert a["z_spec"] == 0.5


def test_sentinel_flux_gives_nan_not_mag225():
    flux = pd.Series([-9999.0, 10.0, 0.0, -1.0])
    mw = pd.Series([-9999.0, 0.9, 0.8, 0.9])
    mag = dered_mag(flux, mw)
    assert np.isnan(mag.iloc[0])          # the (-9999)/(-9999)=+1 trap
    assert abs(mag.iloc[1] - (22.5 - 2.5 * np.log10(10.0 / 0.9))) < 1e-6
    assert np.isnan(mag.iloc[2]) and np.isnan(mag.iloc[3])


def test_sentinel_transmission_gives_nan():
    flux = pd.Series([10.0, 10.0])
    mw = pd.Series([0.0, 5.0])            # zero and unphysical > 1
    mag = dered_mag(flux, mw)
    assert mag.isna().all()
