#!/usr/bin/env python
"""Pantheon+ Hubble diagram prep (capstone Phase 1).

Downloads the Pantheon+SH0ES release (Scolnic et al. 2022; Brout et al.
2022) and builds the HD in the established survey schema. Unlike Union3,
per-SN standardized distances ARE released (m_b_corr; the constant M is
profiled downstream), along with the full 1701x1701 stat+syst covariance
used by the GLS fit (their README explicitly warns the diagonal errors
are for plotting only -- our GLS-vs-diagonal comparison quantifies that).

Duplicates: 1701 rows cover ~1550 unique SNe (same SN, multiple
surveys). `row_index` preserves covariance addressing; the kappa run
uses one row per CID (identical sightline), and fits re-expand.

cluster_targeted: HST cluster-search SNe flagged by positional match
(< 3') to the Union3 See-Change (SuzukiRubin) sightlines, plus any
IDSURVEY=101 (SNAP/SCP) SNe at z > 0.9 (the Suzuki-type cluster-survey
regime).

Run: .venv/bin/python scripts/pantheon_prep.py
Output: output/pantheon/pantheon_hd.csv, data_pantheon/{dat,cov,MANIFEST}
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.kappa import angular_sep_arcsec

BASE = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
        "main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/")
DATA = Path("data_pantheon")
OUT = Path("output/pantheon/pantheon_hd.csv")

SURVEY = {1: "SDSS", 4: "SNLS", 5: "CSP", 10: "DES", 15: "PS1MD",
          18: "CNIa0.02", 50: "LOWZ/JRK07", 51: "LOSS1", 56: "SOUSA",
          57: "LOSS2", 61: "CFA1", 62: "CFA2", 63: "CFA3S", 64: "CFA3K",
          65: "CFA4p2", 66: "CFA4p3", 100: "HST", 101: "SNAP",
          106: "CANDELS", 150: "FOUND"}  # release README, 4_DISTANCES


def fetch():
    DATA.mkdir(exist_ok=True)
    import datetime
    lines = []
    for fn in ("Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"):
        dest = DATA / fn
        if not dest.exists():
            urlretrieve(BASE + fn.replace("+", "%2B"), dest)
        lines.append(f"{fn} from PantheonPlusSH0ES/DataRelease "
                     f"{datetime.datetime.utcnow().isoformat()[:16]}Z "
                     f"({dest.stat().st_size} bytes)")
    (DATA / "MANIFEST.txt").write_text("\n".join(lines) + "\n")


def main():
    fetch()
    d = pd.read_csv(DATA / "Pantheon+SH0ES.dat", sep=r"\s+")
    print(f"{len(d)} rows, {d.CID.nunique()} unique SNe")

    ra = np.where(d.HOST_RA.to_numpy() > -100, d.HOST_RA.to_numpy(),
                  d.RA.to_numpy())
    dec = np.where(d.HOST_DEC.to_numpy() > -100, d.HOST_DEC.to_numpy(),
                   d.DEC.to_numpy())
    out = pd.DataFrame({
        "CID": d.CID, "row_index": np.arange(len(d)),
        "sample": d.IDSURVEY.map(SURVEY).fillna("other"),
        "HOST_RA": ra, "HOST_DEC": dec, "zHD": d.zHD,
        "MU": d.m_b_corr, "MUERR": d.m_b_corr_err_DIAG,
        "HOST_LOGMASS": np.nan, "PROBIA": 1.0, "clipped": False,
        "is_calibrator": d.IS_CALIBRATOR.astype(bool),
    })
    out = out[~out.is_calibrator].reset_index(drop=True)

    # cluster-targeted: positional match to Union3 See-Change + SNAP z>0.9
    u3 = pd.read_csv("output/union3/union3_hd.csv")
    sc = u3[u3.cluster_targeted]
    flag = np.zeros(len(out), dtype=bool)
    for _, r in sc.iterrows():
        sep = angular_sep_arcsec(r.HOST_RA, r.HOST_DEC,
                                 out.HOST_RA.to_numpy(),
                                 out.HOST_DEC.to_numpy())
        flag |= sep < 180.0
    flag |= (out["sample"] == "SNAP") & (out.zHD > 0.9)
    out["cluster_targeted"] = flag
    print(f"cluster-targeted flagged: {int(flag.sum())} "
          f"({int(((out['sample']=='SNAP') & (out.zHD>0.9)).sum())} SNAP "
          f"z>0.9)")
    n_z = int(((out.zHD > 0.1) & (out.zHD < 1.58)).sum())
    print(f"z in (0.1, 1.58): {n_z} rows, "
          f"{out[(out.zHD>0.1)&(out.zHD<1.58)].CID.nunique()} unique")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
