#!/usr/bin/env python
"""Joint three-compilation latent-EIV amplitude (ApJ Phase 1c capstone).

Stacks the per-SN classical/Berkson noise decompositions of all three
compilations on the maximal-independent sightline set (best-measured
representative per shared sightline, as in joint_fit.py) and fits a
single latent amplitude with survey-specific z-bin priors (tau_b
estimated per compilation x bin).

Run after latent_fit.py --noise for all surveys:
    .venv/bin/python scripts/joint_latent.py
Output: output/joint/joint_latent.json (tracked)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa.latent import fit_amplitude
from latent_fit import load_merged, prep_arrays, survey_paths
from joint_fit import match_pairs

OUT = Path("output/joint/joint_latent.json")


def main():
    rng = np.random.default_rng(31415)
    frames = []
    for survey, tag in (("des", "DES"), ("union3", "Union3"),
                        ("pantheon", "PantheonP")):
        kcsv, ncsv, _ = survey_paths(survey)
        d = load_merged(kcsv, noise_csv=ncsv)
        x, y, zbin, s2c, s2b, sig2y, _ = prep_arrays(d, afid=0.79)
        f = pd.DataFrame({"x": x, "y": y, "s2c": s2c, "s2b": s2b,
                          "sig2y": sig2y,
                          "zbin": [f"{tag}|{z}" for z in zbin],
                          "HOST_RA": d.HOST_RA, "HOST_DEC": d.HOST_DEC,
                          "zHD": d.zHD, "MUERR": d.MUERR,
                          "compilation": tag})
        frames.append(f)
        print(f"{tag}: {len(f)} SNe with noise decompositions")
    d = pd.concat(frames, ignore_index=True)

    # dedup: best-measured representative per shared sightline
    n = len(d)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for ca, cb in (("DES", "Union3"), ("Union3", "PantheonP"),
                   ("DES", "PantheonP")):
        for i, j in match_pairs(d, ca, cb):
            parent[find(i)] = find(j)
    comp = np.array([find(i) for i in range(n)])
    sig = d.MUERR.to_numpy()
    rep = []
    for u in np.unique(comp):
        m = np.flatnonzero(comp == u)
        rep.append(m[np.argmin(sig[m])])
    r = d.iloc[rep].reset_index(drop=True)
    print(f"unique sightlines: {len(r)}")

    fit = fit_amplitude(r.x.to_numpy(), r.s2c.to_numpy(),
                        r.zbin.to_numpy(), r.y.to_numpy(),
                        r.sig2y.to_numpy(), rng=rng, n_boot=200,
                        s2_berk=r.s2b.to_numpy())
    print(f"JOINT latent EIV: A = {fit['A']:.3f} +- {fit['A_err']:.3f} "
          f"(post {fit['A_err_post']:.3f}, boot {fit['A_err_boot']:.3f}; "
          f"<w> = {fit['mean_reliability']:.2f})")
    OUT.write_text(json.dumps({"n_unique": len(r), "joint_latent": fit},
                              indent=2, default=float))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
