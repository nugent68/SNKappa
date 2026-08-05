#!/usr/bin/env python
"""Per-galaxy FastSpecFit masses (and velocity dispersions) for the
spec-z foregrounds — the spectroscopic mass variant.

The pipeline uses DESI spectra only for redshifts; masses are always
photometric (recalibrated TO the FSF scale, but never per-galaxy FSF).
This closes that gap for the ~spec-z foreground subset:

  --stage map    (laptop) rebuild the ls_id <-> TARGETID association
                 for every region's spec-matched galaxies (the cached
                 zpix tables carry TARGETID; clean_and_merge's 1-arcsec
                 closest-fiber match is reproduced here) ->
                 data_nir/fsf_targetid_map.csv
  [Perlmutter]   scripts/frankenblast/fsf_extract.py pulls LOGMSTAR +
                 VDISP for those TARGETIDs from the public DR1 VAC ->
                 fsf_extract.csv
  --stage join   (laptop) join map + extraction into the override
                 table data_nir/fsf_masses_override.csv with column
                 logm_p50 (the --fb-masses convention), plus vdisp for
                 the halo-rung check (fsf_vdisp.py).

Run: .venv/bin/python scripts/fsf_masses.py --stage map
     scp + run extraction on Perlmutter (see fsf_extract.py)
     .venv/bin/python scripts/fsf_masses.py --stage join
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa import catalog
from snkappa.datalab import TapClient
from snkappa.kappa import angular_sep_arcsec
import des_full
import union3_full as uf

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


MAP_CSV = Path("data_nir/fsf_targetid_map.csv")
EXTRACT_CSV = Path("data_nir/fsf_extract.csv")
OUT_CSV = Path("data_nir/fsf_masses_override.csv")
MIN_DCHI2 = catalog.MIN_DELTACHI2


def region_map(cfg, tap, rows):
    """Reproduce the clean_and_merge spec crossmatch, keeping TARGETID."""
    df, zpix = catalog.fetch_regional(cfg, tap)
    df = catalog.clean_and_merge(cfg, df, zpix)
    zp = zpix.copy()
    prim = zp["zcat_primary"].astype(str).str.lower().isin(
        ["t", "true", "1"])
    zp = zp[prim & (zp["coadd_fiberstatus"] == 0)
            & (zp["deltachi2"] > MIN_DCHI2)]
    spec = df[df.z_spec.notna()]
    if not len(spec) or not len(zp):
        return
    c_gal = SkyCoord(spec.ra.to_numpy() * u.deg,
                     spec.dec.to_numpy() * u.deg)
    c_fib = SkyCoord(zp.mean_fiber_ra.to_numpy() * u.deg,
                     zp.mean_fiber_dec.to_numpy() * u.deg)
    idx, sep, _ = c_gal.match_to_catalog_sky(c_fib)
    good = sep.arcsec < 1.0
    for ls, ti, zs in zip(spec.ls_id.to_numpy()[good],
                          zp.targetid.to_numpy()[idx][good],
                          spec.z_spec.to_numpy()[good]):
        rows.append({"ls_id": int(ls), "targetid": int(ti),
                     "z_spec": float(zs)})


def stage_map():
    rows = []
    args = SimpleNamespace(n_rand=500, smhm_inverse="posterior",
                           logmh_max=13.8)
    res = pd.read_csv("output/des_full/des_all_kappa.csv")
    for gname in ("X", "S", "C", "E"):
        gsn = res[res.GROUP == gname]
        fields, center = des_full.FIELD_GROUPS[gname]
        rad = angular_sep_arcsec(center[0], center[1],
                                 gsn.HOST_RA.to_numpy(),
                                 gsn.HOST_DEC.to_numpy()).max() / 3600.0
        cfg = des_full.make_cfg(args, center, rad + 0.25)
        tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
        n0 = len(rows)
        region_map(cfg, tap, rows)
        log(f"des:{gname}: +{len(rows)-n0} spec matches")
    for survey in ("union3", "pantheon"):
        reg = pd.read_csv(f"output/{survey}/regions.csv")
        for _, rrow in reg[reg.process].iterrows():
            cfg = uf.make_cfg((rrow.ra, rrow.dec),
                              max(rrow.radius_deg + 0.25, 0.45), 200)
            tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
            try:
                region_map(cfg, tap, rows)
            except Exception as exc:
                log(f"  {survey}:{rrow.region} skipped ({exc})")
        log(f"{survey}: total {len(rows)} matches")
    m = pd.DataFrame(rows).drop_duplicates("ls_id", keep="first")
    m.to_csv(MAP_CSV, index=False)
    log(f"saved {MAP_CSV}: {len(m)} unique galaxies, "
        f"{m.targetid.nunique()} unique targetids")


def stage_join():
    m = pd.read_csv(MAP_CSV)
    e = pd.read_csv(EXTRACT_CSV)
    j = m.merge(e, on="targetid", how="inner")
    j = j[np.isfinite(j.logmstar_fsf) & (j.logmstar_fsf > 6)]
    out = pd.DataFrame({"ls_id": j.ls_id, "logm_p50": j.logmstar_fsf,
                        "vdisp": j.vdisp, "z_spec": j.z_spec})
    out = out.drop_duplicates("ls_id", keep="first")
    out.to_csv(OUT_CSV, index=False)
    log(f"saved {OUT_CSV}: {len(out)} galaxies with FSF masses "
        f"({int((out.vdisp > 0).sum())} with vdisp)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("map", "join"))
    a = ap.parse_args()
    stage_map() if a.stage == "map" else stage_join()


if __name__ == "__main__":
    main()
