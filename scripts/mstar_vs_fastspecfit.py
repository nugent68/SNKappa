#!/usr/bin/env python
"""Cross-check the Nir1um stellar-mass scale against the DESI DR1
FastSpecFit VAC for the exact lens samples used in the DeltaSigma closure.

If our M* runs low for massive red galaxies, the low predicted DeltaSigma
amplitude reflects the M* scale (input), not the SMHM/profile chain. If the
scales agree, the discrepancy sits in the halo chain (or satellites).

Run on Perlmutter: ./venv/bin/python scripts/mstar_vs_fastspecfit.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snkappa.stellar import make_estimator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from delta_sigma_closure import BINS, CATDIR, mags_from_fluxes

FSF = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/"
           "v3.0/catalogs")


def load_fsf_logmstar(program):
    """TARGETID -> logM* map from the FastSpecFit healpix catalogs."""
    tids, lms = [], []
    for f in sorted(FSF.glob(f"fastspec-iron-main-{program}-nside1-hp*.fits")):
        with fits.open(f, memmap=True) as h:
            hdu_m = hdu_t = None
            for hd in h[1:]:
                names = getattr(hd, "columns", None)
                if names is None:
                    continue
                if "LOGMSTAR" in names.names and hdu_m is None:
                    hdu_m = hd
                if "TARGETID" in names.names and hdu_t is None:
                    hdu_t = hd
            if hdu_m is None or hdu_t is None:
                raise RuntimeError(f"no LOGMSTAR/TARGETID in {f}")
            tids.append(np.asarray(hdu_t.data["TARGETID"]))
            lms.append(np.asarray(hdu_m.data["LOGMSTAR"], dtype=float))
    return np.concatenate(tids), np.concatenate(lms)


def main():
    rng = np.random.default_rng(7)
    from astropy.cosmology import Planck18 as cosmo
    est = make_estimator("nir1um", cosmo)
    out = {}
    cache = {}
    for sample, zlo, zhi, mcut in BINS:
        program = "bright" if sample.startswith("BGS") else "dark"
        if program not in cache:
            cache[program] = load_fsf_logmstar(program)
        tid_f, lm_f = cache[program]
        order = np.argsort(tid_f)

        t = Table.read(CATDIR / f"{sample}_clustering.dat.fits")
        sel = (np.asarray(t["Z"]) > zlo) & (np.asarray(t["Z"]) < zhi)
        if mcut is not None:
            col = "ABSMAG_RP1" if "ABSMAG_RP1" in t.colnames else "ABSMAG_R"
            sel &= np.asarray(t[col], dtype=float) < mcut
        idx = np.flatnonzero(sel)
        idx = rng.choice(idx, min(20000, idx.size), replace=False)
        t = t[idx]
        z = np.asarray(t["Z"], dtype=float)
        ours = est.logmstar(mags_from_fluxes(t, sample), z)

        tid = np.asarray(t["TARGETID"])
        pos = np.clip(np.searchsorted(tid_f[order], tid), 0, order.size - 1)
        row = order[pos]
        good = (tid_f[row] == tid) & np.isfinite(ours)
        theirs = lm_f[row]
        good &= np.isfinite(theirs) & (theirs > 6)
        d = (ours - theirs)[good]
        out[f"{sample}_{zlo}_{zhi}"] = {
            "n": int(good.sum()),
            "median_offset_dex": float(np.median(d)),
            "nmad_dex": float(1.4826 * np.median(np.abs(d - np.median(d)))),
            "our_med": float(np.median(ours[good])),
            "fsf_med": float(np.median(theirs[good])),
        }
        print(f"{sample} z[{zlo},{zhi}]: N={good.sum()} "
              f"our-FSF median = {np.median(d):+.3f} dex "
              f"(nmad {1.4826*np.median(np.abs(d-np.median(d))):.3f}) "
              f"our_med={np.median(ours[good]):.2f} "
              f"fsf_med={np.median(theirs[good]):.2f}", flush=True)
    Path("output/delta_sigma").mkdir(parents=True, exist_ok=True)
    Path("output/delta_sigma/mstar_vs_fastspecfit.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
