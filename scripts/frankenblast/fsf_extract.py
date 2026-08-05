#!/usr/bin/env python
"""Extract FastSpecFit LOGMSTAR + VDISP for a TARGETID list (Perlmutter).

Reads the public DESI DR1 FastSpecFit VAC (iron v3.0 healpix catalogs)
and writes fsf_extract.csv (targetid, logmstar_fsf, vdisp) for the
targetids in fsf_targetid_map.csv. Loops both programs; memmap keeps
the memory footprint small.

Run on Perlmutter (any login node, ~minutes):
    python3 fsf_extract.py fsf_targetid_map.csv fsf_extract.csv
"""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

FSF = Path("/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/"
           "iron/v3.0/catalogs")


def main():
    map_csv, out_csv = sys.argv[1], sys.argv[2]
    want = set()
    with open(map_csv) as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        ti_col = hdr.index("targetid")
        for line in fh:
            want.add(int(line.split(",")[ti_col]))
    print(f"{len(want)} targetids wanted", flush=True)

    rows = {}
    for program in ("bright", "dark"):
        for f in sorted(FSF.glob(
                f"fastspec-iron-main-{program}-nside1-hp*.fits")):
            with fits.open(f, memmap=True) as h:
                hdu_m = hdu_t = None
                for hd in h[1:]:
                    cols = getattr(hd, "columns", None)
                    if cols is None:
                        continue
                    names = cols.names
                    if "LOGMSTAR" in names and hdu_m is None:
                        hdu_m = hd
                    if "TARGETID" in names and hdu_t is None:
                        hdu_t = hd
                if hdu_m is None or hdu_t is None:
                    continue
                tid = np.asarray(hdu_t.data["TARGETID"])
                sel = np.isin(tid, np.fromiter(want, dtype=np.int64))
                if not sel.any():
                    continue
                lm = np.asarray(hdu_m.data["LOGMSTAR"], float)[sel]
                vnames = hdu_m.columns.names
                vd = (np.asarray(hdu_m.data["VDISP"], float)[sel]
                      if "VDISP" in vnames else np.full(sel.sum(), np.nan))
                for t, m, v in zip(tid[sel], lm, vd):
                    rows[int(t)] = (float(m), float(v))
            print(f"{f.name}: {sel.sum()} matched "
                  f"({len(rows)} total)", flush=True)

    with open(out_csv, "w") as fh:
        fh.write("targetid,logmstar_fsf,vdisp\n")
        for t, (m, v) in rows.items():
            fh.write(f"{t},{m},{v}\n")
    print(f"saved {out_csv}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
