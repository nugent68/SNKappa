"""LOS catalog construction: LS tractor master list + photo-z + DESI spec-z.

Strategy: ONE regional query per table covering the SN aperture plus the whole
random-sightline annulus (radius = annulus_outer + aperture). Randoms are then
drawn locally from the same catalog, guaranteeing an identical estimator and
naturally matched depth, at a fraction of the query load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

from .kappa import angular_sep_arcsec

# maskbits to reject: BRIGHT(1), ALLMASK_G(5), ALLMASK_R(6), ALLMASK_Z(7),
# BAILOUT(10) -- LS DR9/DR10 bit definitions
BAD_MASKBITS = (1 << 1) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 10)

MIN_DELTACHI2 = 25.0  # DESI recommended galaxy-redshift confidence cut

_TRACTOR_COLS = (
    "ls_id, ra, dec, type, maskbits, "
    "flux_g, flux_r, flux_i, flux_z, flux_w1, flux_w2, "
    "flux_ivar_g, flux_ivar_r, flux_ivar_i, flux_ivar_z, "
    "flux_ivar_w1, flux_ivar_w2, "
    "mw_transmission_g, mw_transmission_r, mw_transmission_i, "
    "mw_transmission_z, mw_transmission_w1, mw_transmission_w2, "
    "fracflux_r, fracflux_z, fracmasked_r, fracmasked_z, fracin_r, fracin_z"
)

_PZ_COLS = "z_phot_median, z_phot_std, z_phot_l68, z_phot_u68"


def region_radius_deg(cfg) -> float:
    return cfg.randoms.annulus_deg[1] + cfg.los.aperture_radius_arcmin / 60.0


def dered_mag(flux, mw_transmission):
    """Dereddened AB mag from nanomaggies, robust to Data Lab NULL sentinels.

    Data Lab serves NULL as -9999.0; naively (-9999)/(-9999) = +1.0 would
    fabricate mag=22.5 for bands a survey region never observed (e.g. i-band
    in the LS north). Any nonpositive or sentinel flux/transmission -> NaN.
    """
    flux = flux.where((flux > 0) & (flux > -9000))
    mw = mw_transmission.where((mw_transmission > 0) & (mw_transmission <= 1.0))
    dered = flux / mw
    with np.errstate(divide="ignore", invalid="ignore"):
        return 22.5 - 2.5 * np.log10(dered.where(dered > 0))


def fetch_regional(cfg, tap) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch (tractor master + photo-z coalesced, DESI zpix) for the region."""
    ra, dec = cfg.source.ra_src, cfg.source.dec_src
    radius = region_radius_deg(cfg)
    ls = cfg.data.ls_release
    desi = cfg.data.desi_release

    # server-side flux cut 0.5 mag fainter than the final (dereddened) cut
    flux_min = 10.0 ** ((22.5 - (cfg.los.mag_limit + 0.5)) / 2.5)
    band = cfg.los.mag_limit_band

    cone = f"'t'=Q3C_RADIAL_QUERY(t.ra, t.dec, {ra}, {dec}, {radius})"
    cuts = (f"AND t.brick_primary=1 AND t.type != 'PSF' AND t.type != 'DUP' "
            f"AND t.flux_{band} > {flux_min:.4f}")

    master = tap.query(
        f"SELECT {_TRACTOR_COLS} FROM ls_{ls}.tractor t WHERE {cone} {cuts}",
        label="regional:tractor")

    pz10 = tap.query(
        f"SELECT t.ls_id, {_PZ_COLS} FROM ls_{ls}.tractor t "
        f"JOIN ls_{ls}.photo_z p ON t.ls_id=p.ls_id WHERE {cone} {cuts}",
        label=f"regional:photo_z_{ls}")

    pz9 = tap.query(
        f"SELECT t.ls_id, {_PZ_COLS} FROM ls_dr9.tractor t "
        f"JOIN ls_dr9.photo_z p ON t.ls_id=p.ls_id WHERE {cone} {cuts}",
        label="regional:photo_z_dr9")

    zpix = tap.query(
        "SELECT targetid, mean_fiber_ra, mean_fiber_dec, z, zerr, zwarn, "
        "spectype, zcat_primary, coadd_fiberstatus, deltachi2 "
        f"FROM desi_{desi}.zpix WHERE 't'=Q3C_RADIAL_QUERY("
        f"mean_fiber_ra, mean_fiber_dec, {ra}, {dec}, {radius}) "
        f"AND zwarn=0 AND spectype='GALAXY' "
        f"AND z > 0.001 AND z < {cfg.source.z_src + 0.1}",
        label="regional:zpix")

    # coalesce photo-z: DR10 (south) first, DR9 (north; identical ls_id) second
    pz10 = pz10.rename(columns={c: c for c in pz10.columns})
    pz10["pz_source"] = f"ls_{ls}"
    pz9["pz_source"] = "ls_dr9"
    pz = pd.concat([pz10, pz9]).drop_duplicates("ls_id", keep="first")
    df = master.merge(pz, on="ls_id", how="left")
    return df, zpix


def clean_and_merge(cfg, df: pd.DataFrame, zpix: pd.DataFrame) -> pd.DataFrame:
    """Quality cuts, dereddening, spec-z crossmatch, exclusion flags."""
    # -- quality cuts (local; server already did brick_primary/type/flux) ----
    for b in ("r", "z"):
        df = df[(df[f"fracflux_{b}"] < 0.5) & (df[f"fracmasked_{b}"] < 0.4)
                & (df[f"fracin_{b}"] > 0.3)]
    df = df[(df["maskbits"].astype(int) & BAD_MASKBITS) == 0].copy()

    # -- dereddened AB mags (fluxes are nanomaggies; deredden = /transmission)
    for b in ("g", "r", "i", "z", "w1", "w2"):
        df[f"mag_{b}"] = dered_mag(df[f"flux_{b}"], df[f"mw_transmission_{b}"])
    # require detected W1 (SNR>2) for the NIR mass estimator; else NaN -> the
    # estimator falls back to the optical-color method
    w1_snr = df["flux_w1"] * np.sqrt(df["flux_ivar_w1"].clip(lower=0))
    df.loc[w1_snr < 2.0, "mag_w1"] = np.nan

    band = cfg.los.mag_limit_band
    df = df[df[f"mag_{band}"] <= cfg.los.mag_limit].copy()
    # need g and z for stellar masses
    df = df[np.isfinite(df["mag_g"]) & np.isfinite(df["mag_z"])].copy()

    # -- DESI spec-z crossmatch (<= 1 arcsec) --------------------------------
    zpix = zpix.copy()
    prim = zpix["zcat_primary"].astype(str).str.lower().isin(["t", "true", "1"])
    zpix = zpix[prim & (zpix["coadd_fiberstatus"] == 0)
                & (zpix["deltachi2"] > MIN_DELTACHI2)]
    df["z_spec"] = np.nan
    if len(zpix) and len(df):
        c_gal = SkyCoord(df["ra"].to_numpy() * u.deg, df["dec"].to_numpy() * u.deg)
        c_spec = SkyCoord(zpix["mean_fiber_ra"].to_numpy() * u.deg,
                          zpix["mean_fiber_dec"].to_numpy() * u.deg)
        idx, sep, _ = c_spec.match_to_catalog_sky(c_gal)
        good = sep < 1.0 * u.arcsec
        df.iloc[idx[good], df.columns.get_loc("z_spec")] = \
            zpix["z"].to_numpy()[good]

    # -- redshift bookkeeping ------------------------------------------------
    df = df.rename(columns={"z_phot_median": "zp_med", "z_phot_std": "zp_std",
                            "z_phot_l68": "zp_l68", "z_phot_u68": "zp_u68"})
    # a spec-z at/above z_src means "confirmed background": remove entirely
    # (do NOT fall back to photo-z, which could wrongly place it foreground)
    df = df[~(df["z_spec"] >= cfg.source.z_src)].copy()
    has_spec = df["z_spec"].notna()
    has_phot = df["zp_med"].notna() & (df["zp_med"] > 0)
    n_dropped = int((~has_spec & ~has_phot).sum())
    df = df[has_spec | has_phot].copy()

    # -- deflector and group exclusion flags ---------------------------------
    dfl = cfg.deflector
    sep_lens = angular_sep_arcsec(dfl.ra_lens, dfl.dec_lens,
                                  df["ra"].to_numpy(), df["dec"].to_numpy())
    df["sep_lens_arcsec"] = sep_lens
    df["excl_deflector"] = sep_lens < dfl.r_exclude_arcsec

    grp = cfg.lens_group
    zl, dz = dfl.z_lens, grp.dz_group
    in_r = sep_lens < grp.r_group_arcmin * 60.0
    spec_member = df["z_spec"].notna() & (np.abs(df["z_spec"] - zl) < dz)
    sigma_eff = 0.5 * ((df["zp_med"] - df["zp_l68"]).clip(lower=0.01)
                       + (df["zp_u68"] - df["zp_med"]).clip(lower=0.01))
    phot_member = (df["z_spec"].isna()
                   & (np.abs(df["zp_med"] - zl) < np.maximum(dz, sigma_eff)))
    df["is_group"] = in_r & (spec_member | phot_member) & ~df["excl_deflector"]

    df = df.reset_index(drop=True)
    df.attrs["n_dropped_no_z"] = n_dropped
    return df


def area_fraction(cfg, df: pd.DataFrame, ra0: float, dec0: float,
                  n_az: int = 8, n_rad: int = 5) -> tuple[float, int]:
    """Heuristic unmasked-area fraction of an aperture: compare source counts
    in equal-area cells; cells below 30% of the median density flag masking or
    a survey edge. Returns (fraction_ok, n_cells_flagged)."""
    r_ap = cfg.los.aperture_radius_arcmin * 60.0
    theta = angular_sep_arcsec(ra0, dec0, df["ra"].to_numpy(),
                               df["dec"].to_numpy())
    sel = theta < r_ap
    if sel.sum() < n_az * n_rad:
        return 0.0, n_az * n_rad
    ddec = df["dec"].to_numpy()[sel] - dec0
    dra = ((df["ra"].to_numpy()[sel] - ra0)
           * np.cos(np.radians(dec0)))
    az = np.arctan2(ddec, dra)
    r_edges = r_ap * np.sqrt(np.linspace(0, 1, n_rad + 1))  # equal-area rings
    az_bins = np.digitize(az, np.linspace(-np.pi, np.pi, n_az + 1)) - 1
    r_bins = np.digitize(theta[sel], r_edges) - 1
    counts = np.zeros((n_rad, n_az))
    for rb, ab in zip(np.clip(r_bins, 0, n_rad - 1),
                      np.clip(az_bins, 0, n_az - 1)):
        counts[rb, ab] += 1
    med = np.median(counts)
    flagged = int((counts < 0.3 * med).sum())
    return 1.0 - flagged / counts.size, flagged
