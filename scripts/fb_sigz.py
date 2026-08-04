#!/usr/bin/env python
"""Join per-galaxy photo-z widths onto the FB target list (close-out P1).

The FB zfix runs fix each galaxy's redshift at z_best, so their mass
posteriors omit the photo-z uncertainty for the ~83% of targets whose
z_best is a DECaLS photo-z. This script recovers each target's sigma_z
from the regional catalogs (warm TAP cache; same driver code paths as
fb_targets.py) so the +/-sigma_z refits can propagate it.

sigma_z = max(zp_std, (zp_u68 - zp_l68)/2), 0 for spec-z targets.

Output: data_nir/fb_targets_sigz.csv (ls_id, z_best, z_is_spec, sig_z)

Run: .venv/bin/python scripts/fb_sigz.py
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

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


OUT = Path("data_nir/fb_targets_sigz.csv")
TARGETS = Path("data_nir/fb_targets_all.csv")


def sigz_frame(df):
    """ls_id -> sigma_z for one regional catalog."""
    std = df.zp_std.to_numpy(float)
    half68 = 0.5 * (df.zp_u68.to_numpy(float) - df.zp_l68.to_numpy(float))
    sig = np.nanmax(np.vstack([std, half68]), axis=0)
    return pd.DataFrame({"ls_id": df.ls_id.astype(str), "sig_z": sig})


def main():
    tg = pd.read_csv(TARGETS)
    tg["ls_id"] = tg.ls_id.astype(str)
    need = set(tg.ls_id)
    frames = []

    args = SimpleNamespace(n_rand=500, smhm_inverse="posterior",
                           logmh_max=13.8)
    res = pd.read_csv("output/nir_video/des_all_kappa_nir.csv")
    for gname in ("X", "C", "E"):
        gsn = res[res.GROUP == gname]
        fields, center = des_full.FIELD_GROUPS[gname]
        rad = angular_sep_arcsec(center[0], center[1],
                                 gsn.HOST_RA.to_numpy(),
                                 gsn.HOST_DEC.to_numpy()).max() / 3600.0
        cfg = des_full.make_cfg(args, center, rad + 0.25)
        tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
        df = catalog.clean_and_merge(cfg, *catalog.fetch_regional(cfg, tap))
        frames.append(sigz_frame(df))
        log(f"des:{gname}: {len(df)} galaxies scanned")

    for survey in ("union3", "pantheon"):
        reg = pd.read_csv(f"output/nir_video/{survey}/regions.csv")
        for _, rrow in reg[reg.process].iterrows():
            cfg = uf.make_cfg((rrow.ra, rrow.dec),
                              max(rrow.radius_deg + 0.25, 0.45), 200)
            tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
            df = catalog.clean_and_merge(
                cfg, *catalog.fetch_regional(cfg, tap))
            frames.append(sigz_frame(df))
        log(f"{survey}: regions scanned")

    sz = pd.concat(frames, ignore_index=True)
    sz = sz[sz.ls_id.isin(need)].drop_duplicates("ls_id", keep="first")
    out = tg[["ls_id", "z_best", "z_is_spec"]].merge(sz, on="ls_id",
                                                     how="left")
    out.loc[out.z_is_spec.astype(bool), "sig_z"] = 0.0
    n_missing = int(out.sig_z.isna().sum())
    # any stragglers (shouldn't happen): fall back to the LS scaling
    out["sig_z"] = out.sig_z.fillna(0.03 * (1.0 + out.z_best))
    out.to_csv(OUT, index=False)
    phot = out[~out.z_is_spec.astype(bool)]
    log(f"saved {OUT}: {len(out)} rows, {n_missing} fallback; "
        f"photo-z sig_z median {phot.sig_z.median():.3f}, "
        f"p90 {phot.sig_z.quantile(0.9):.3f}")


if __name__ == "__main__":
    main()
