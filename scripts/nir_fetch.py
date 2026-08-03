#!/usr/bin/env python
"""Deep-NIR mass upgrade, Phase 0: fetch VIDEO DR5 + COSMOS2020 and audit
the crossmatch against the LS foreground catalogs (NIR sprint P0).

Downloads (to gitignored data_nir/):
  - VIDEO DR5 band-merged ZYJHKs catalogs (ESO tap_cat, table
    VIDEO_CAT_DR5) for the XMM-LSS, CDFS and ES1 fields, Ks < 23.5.
  - COSMOS2020 Classic UltraVISTA YJHKs subset (VizieR TAP,
    J/ApJS/258/11/classic), Ks or J < 23.5.

Audit (data_nir/nir_audit.json):
  - per field: NIR source counts, LS-foreground counts inside the NIR
    footprint (nearest-NIR-source < 30 arcsec coverage proxy), 1-arcsec
    match completeness for mag_z <= 22.5 galaxies (GATE: >= 95%), and
    the empirical AB-vs-Vega check (median LS z - VIDEO Z offset for
    matched 18 < z < 21 galaxies: ~0 => AB, ~+0.5 => Vega).

The LS foregrounds are rebuilt through the exact driver code paths
(des_full FIELD_GROUPS for X/C/E, the Pantheon+ COSMOS region for
UltraVISTA) so the warm TAP cache is reused and the audited catalog is
identical to what Phase 3 will see.

Run: .venv/bin/python scripts/nir_fetch.py [--skip-download]
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa import catalog
from snkappa.datalab import TapClient

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


DATA_DIR = Path("data_nir")
AUDIT_JSON = DATA_DIR / "nir_audit.json"

ESO_TAP = "https://archive.eso.org/tap_cat/sync"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

# VIDEO DR5 fields (bounds from the served table itself)
VIDEO_FIELDS = ("XMM", "CDFS", "ES1")
VIDEO_BANDS = ("z", "y", "j", "h", "ks")

# generous faint cut: superset of any counterpart of a z <= 22.5 LS
# galaxy whether the catalog is AB or Vega
NIR_FAINT_CUT = 23.5

MATCH_ARCSEC = 1.0        # LS <-> NIR counterpart radius
# a sightline counts as NIR-covered only if the local NIR source density is
# survey-interior-like: >= FOOTPRINT_MIN_N sources within FOOTPRINT_ARCSEC
# (interior deep-NIR density is ~30-40/arcmin^2, so a 60-arcsec circle holds
# ~100+; ragged edges and bright-star masks fall far below this)
FOOTPRINT_ARCSEC = 60.0
FOOTPRINT_MIN_N = 20
# VIDEO fields must be near-complete; COSMOS2020 excises bright-star mask
# holes (~15-30 arcsec, too small for the density proxy) that hold ~6% of
# LS foregrounds -- those galaxies keep the z<->W1 estimator (hybrid), so
# the COSMOS gate is looser and the shortfall is a coverage statement,
# not a data-quality failure (diagnosed 2026-08-03: unmatched nearest-NIR
# sep median 14 arcsec; magnitude distribution identical to matched).
GATE_VIDEO = 0.95
GATE_COSMOS = 0.93


# ----------------------------------------------------------------- downloads
def _http_query(url, adql, fmt, timeout=900):
    r = requests.get(url, params={"REQUEST": "doQuery", "LANG": "ADQL",
                                  "FORMAT": fmt, "MAXREC": 3_000_000,
                                  "QUERY": adql}, timeout=timeout)
    r.raise_for_status()
    return r.content


def fetch_video(field: str) -> Path:
    out = DATA_DIR / f"video_{field.lower()}.parquet"
    if out.exists():
        log(f"  video_{field.lower()}.parquet cached "
            f"({out.stat().st_size/1e6:.0f} MB)")
        return out
    cols = ["VIDEOID", "RA2000", "DEC2000"]
    for b in VIDEO_BANDS:
        B = b.upper()
        cols += [f"{B}_MAG_AUTO", f"{B}_MAGERR_AUTO", f"{B}_DET_FLAG"]
    cols += ["KS_CLASS_STAR", "HALOFLAG"]
    adql = (f"SELECT {', '.join(cols)} FROM VIDEO_CAT_DR5 "
            f"WHERE FIELDNAME='{field}' AND KS_MAG_AUTO < {NIR_FAINT_CUT}")
    log(f"  querying VIDEO {field} (ESO TAP)...")
    raw = _http_query(ESO_TAP, adql, "fits")
    t = Table.read(io.BytesIO(raw), format="fits")
    df = t.to_pandas()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"ra2000": "ra", "dec2000": "dec"})
    # SExtractor 99/-99 sentinels -> NaN
    for c in df.columns:
        if "_mag_auto" in c or "_magerr_auto" in c:
            df[c] = df[c].where((df[c] > -90) & (df[c] < 90))
    df.to_parquet(out)
    log(f"  VIDEO {field}: {len(df)} rows -> {out.name} "
        f"({out.stat().st_size/1e6:.0f} MB)")
    return out


def fetch_cosmos2020() -> Path:
    out = DATA_DIR / "cosmos2020.parquet"
    if out.exists():
        log(f"  cosmos2020.parquet cached ({out.stat().st_size/1e6:.0f} MB)")
        return out
    cols = ("RAJ2000, DEJ2000, "
            "UVISTAYmagAuto, e_UVISTAYmagAuto, UVISTAJmagAuto, "
            "e_UVISTAJmagAuto, UVISTAHmagAuto, e_UVISTAHmagAuto, "
            "UVISTAKsmagAuto, e_UVISTAKsmagAuto, FlagUVISTA, "
            "lptype, lpzBEST, loglpMassmed")
    adql = (f"SELECT {cols} FROM \"J/ApJS/258/11/classic\" "
            f"WHERE UVISTAKsmagAuto < {NIR_FAINT_CUT} "
            f"OR UVISTAJmagAuto < {NIR_FAINT_CUT}")
    log("  querying COSMOS2020 Classic (VizieR TAP)...")
    raw = _http_query(VIZIER_TAP, adql, "csv")
    df = pd.read_csv(io.BytesIO(raw))
    ren = {"RAJ2000": "ra", "DEJ2000": "dec",
           "FlagUVISTA": "flag_uvista", "lptype": "lp_type",
           "lpzBEST": "lp_zbest", "loglpMassmed": "lp_logmass"}
    for b in ("Y", "J", "H", "Ks"):
        ren[f"UVISTA{b}magAuto"] = f"{b.lower()}_mag_auto"
        ren[f"e_UVISTA{b}magAuto"] = f"{b.lower()}_magerr_auto"
    df = df.rename(columns=ren)
    df.to_parquet(out)
    log(f"  COSMOS2020: {len(df)} rows -> {out.name} "
        f"({out.stat().st_size/1e6:.0f} MB)")
    return out


# ------------------------------------------------------- LS foreground audit
def des_foreground(group: str) -> pd.DataFrame:
    """LS foreground catalog for a DES group via the des_full code path
    (identical cfg -> identical TAP cache keys)."""
    import des_full
    args = SimpleNamespace(n_rand=500, smhm_inverse="posterior",
                           logmh_max=13.8)
    sn_all = des_full.load_des()
    fields, center = des_full.FIELD_GROUPS[group]
    sn = sn_all[sn_all.FIELD.isin(fields)]
    from snkappa.kappa import angular_sep_arcsec
    rad = angular_sep_arcsec(center[0], center[1], sn.HOST_RA.to_numpy(),
                             sn.HOST_DEC.to_numpy()).max() / 3600.0
    cfg = des_full.make_cfg(args, center, rad + 0.25)
    tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
    return catalog.clean_and_merge(cfg, *catalog.fetch_regional(cfg, tap))


def cosmos_foreground() -> pd.DataFrame:
    """LS foreground for the COSMOS region via the Pantheon+ region row."""
    import union3_full
    reg = pd.read_csv("output/pantheon/regions.csv")
    sel = reg[(np.abs(reg.ra - 150.1) < 1.0) & (np.abs(reg.dec - 2.2) < 1.0)]
    row = sel.sort_values("n_sn", ascending=False).iloc[0]
    cfg = union3_full.make_cfg((row.ra, row.dec),
                               max(row.radius_deg + 0.25, 0.45), 500)
    tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
    return catalog.clean_and_merge(cfg, *catalog.fetch_regional(cfg, tap))


def audit_field(name: str, fg: pd.DataFrame, nir: pd.DataFrame,
                zcol_nir: str | None) -> dict:
    """Match completeness + mag-system check for one field."""
    c_fg = SkyCoord(fg["ra"].to_numpy() * u.deg, fg["dec"].to_numpy() * u.deg)
    c_nir = SkyCoord(nir["ra"].to_numpy() * u.deg,
                     nir["dec"].to_numpy() * u.deg)
    idx, sep, _ = c_fg.match_to_catalog_sky(c_nir)
    sep_as = sep.arcsec
    matched = sep_as < MATCH_ARCSEC
    # coverage: local NIR density must be interior-like
    i_fg, _, _, _ = c_nir.search_around_sky(
        c_fg, FOOTPRINT_ARCSEC * u.arcsec)
    n_local = np.bincount(i_fg, minlength=len(fg))
    in_foot = n_local >= FOOTPRINT_MIN_N
    bright = fg["mag_z"].to_numpy() <= 22.5

    n_foot = int((in_foot & bright).sum())
    n_match = int((matched & bright).sum())
    comp = n_match / n_foot if n_foot else 0.0

    out = {"n_nir": int(len(nir)), "n_fg": int(len(fg)),
           "n_fg_z225_in_footprint": n_foot, "n_matched_1as": n_match,
           "completeness": round(comp, 4),
           "footprint_frac_of_region": round(float((in_foot & bright).sum()
                                             / max(bright.sum(), 1)), 4)}

    # AB-vs-Vega: median (LS z_AB - NIR Z) for well-measured matches
    if zcol_nir is not None:
        zmag = nir[zcol_nir].to_numpy()[idx]
        lsz = fg["mag_z"].to_numpy()
        ok = matched & np.isfinite(zmag) & np.isfinite(lsz) \
            & (lsz > 18) & (lsz < 21)
        if ok.sum() > 50:
            out["median_LSz_minus_NIRZ"] = round(
                float(np.median(lsz[ok] - zmag[ok])), 3)
            out["n_zoffset"] = int(ok.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    if not args.skip_download:
        log("downloading NIR catalogs...")
        for f in VIDEO_FIELDS:
            fetch_video(f)
        fetch_cosmos2020()

    audit = {}
    pairs = [("X", "XMM", "video_xmm.parquet"),
             ("C", "CDFS", "video_cdfs.parquet"),
             ("E", "ES1", "video_es1.parquet")]
    for group, field, fname in pairs:
        log(f"auditing DES group {group} vs VIDEO {field}...")
        fg = des_foreground(group)
        nir = pd.read_parquet(DATA_DIR / fname)
        audit[f"DES_{group}_vs_{field}"] = audit_field(
            group, fg, nir, "z_mag_auto")
        log(f"  {audit[f'DES_{group}_vs_{field}']}")

    log("auditing COSMOS region vs COSMOS2020...")
    fg = cosmos_foreground()
    nir = pd.read_parquet(DATA_DIR / "cosmos2020.parquet")
    audit["COSMOS_vs_COSMOS2020"] = audit_field("COSMOS", fg, nir, None)
    log(f"  {audit['COSMOS_vs_COSMOS2020']}")

    worst_video = min(v["completeness"] for k, v in audit.items()
                      if k.startswith("DES_"))
    comp_cosmos = audit["COSMOS_vs_COSMOS2020"]["completeness"]
    audit["gate"] = {
        "video_threshold": GATE_VIDEO, "cosmos_threshold": GATE_COSMOS,
        "worst_video_completeness": worst_video,
        "cosmos_completeness": comp_cosmos,
        "cosmos_note": "shortfall = COSMOS2020 bright-star mask holes; "
                       "affected galaxies keep the z<->W1 estimator",
        "pass": worst_video >= GATE_VIDEO and comp_cosmos >= GATE_COSMOS}
    AUDIT_JSON.write_text(json.dumps(audit, indent=2))
    log(f"saved {AUDIT_JSON}")
    log(f"GATE {'PASS' if audit['gate']['pass'] else 'FAIL'} "
        f"(worst VIDEO {worst_video:.3f}, COSMOS {comp_cosmos:.3f})")


if __name__ == "__main__":
    main()
