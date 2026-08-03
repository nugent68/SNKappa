#!/usr/bin/env python
"""Stats parity for Union3/Pantheon+ (ApJ Phase 1b): 0.5-deg spatial
block bootstrap and leave-one-sample-out jackknife, mirroring the DES
treatment (permutation nulls already live in the survey fit summaries).

Run: .venv/bin/python scripts/compilation_stats.py
Output: output/{union3,pantheon}/extra_stats.json (tracked)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.fitting import bootstrap_slope

BLOCK_DEG = 0.5


def main():
    rng = np.random.default_rng(2029)
    for survey in ("union3", "pantheon"):
        p = Path(f"output/{survey}/{survey}_kappa.csv")
        d = pd.read_csv(p)
        d = d[~d.clipped.astype(bool)
              & ~d.cluster_targeted.astype(bool)].reset_index(drop=True)
        x, y = d.kappa_ext.to_numpy(), d.hr.to_numpy()
        sig = d.MUERR.to_numpy()
        b0 = float(np.polyfit(x, y, 1, w=1.0 / sig)[0])

        bra = np.floor(d.HOST_RA.to_numpy() / BLOCK_DEG)
        bdec = np.floor(d.HOST_DEC.to_numpy() / BLOCK_DEG)
        bid = pd.factorize(bra * 10000 + bdec)[0]
        blocks = [np.flatnonzero(bid == u) for u in np.unique(bid)]
        bb = np.empty(4000)
        for k in range(4000):
            pick = rng.integers(0, len(blocks), len(blocks))
            idx = np.concatenate([blocks[j] for j in pick])
            bb[k] = np.polyfit(x[idx], y[idx], 1, w=1.0 / sig[idx])[0]

        jack = {}
        for s in d["sample"].unique():
            m = d["sample"] != s
            if m.sum() < 100 or (~m).sum() < 20:
                continue
            bj, ej = bootstrap_slope(x[m], y[m], sig[m], rng, n_boot=500)
            jack[s] = [bj, ej, int((~m).sum())]

        out = {"slope": b0,
               "block_bootstrap": {"block_deg": BLOCK_DEG,
                                   "n_blocks": len(blocks),
                                   "slope_err": float(bb.std())},
               "jackknife_drop_sample": jack}
        dst = Path(f"output/{survey}/extra_stats.json")
        dst.write_text(json.dumps(out, indent=2, default=float))
        jr = {k: round(v[0], 2) for k, v in jack.items()}
        print(f"{survey}: slope {b0:+.2f} | block-boot err "
              f"{bb.std():.2f} ({len(blocks)} blocks) | drop-sample "
              f"slopes {jr}")
        print(f"  saved {dst}")


if __name__ == "__main__":
    main()
