#!/usr/bin/env python
"""Tracked summary of the DESI DR1 (Lensing Without Borders v1.5)
closure comparison — derived amplitude ratios only.

The underlying measured DeltaSigma vectors are DESI-internal, used and
quoted here in advance of their public release by permission of
S. Heydenreich / the DESI Lensing Working Group; the raw vectors are
deliberately NOT tracked (they live in the untracked local
notes/desi_internal/, populated from
/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing on NERSC by
scripts/delta_sigma_closure.py). This script distills the per-bin,
per-survey one-halo amplitude ratios A_DS = <ds_meas ds_pred>/<ds_pred^2>
that the manuscript quotes, for the fiducial (nir1um_fsf + posterior
inversion) chain and the pre-recalibration constant-M*/L variant.

Run: .venv/bin/python scripts/desi_closure_summary.py
Output: output/delta_sigma/desi_closure_summary.json (tracked)
"""

import json
from pathlib import Path

SRC = Path("notes/desi_internal")
OUT = Path("output/delta_sigma/desi_closure_summary.json")
VARIANTS = {"fiducial_nir1um_fsf": "delta_sigma_closure_recal.json",
            "prerecal_constant_ML": "delta_sigma_closure.json"}


def main():
    out = {"provenance": (
        "DESI DR1 Lensing Without Borders v1.5 measured DeltaSigma "
        "(Heydenreich et al. 2025, arXiv:2506.21677), used in advance "
        "of public release by permission of S. Heydenreich / DESI "
        "Lensing WG; one-halo amplitude ratios only -- raw vectors "
        "not distributed. Prediction: SNKappa halo chain "
        "(scripts/delta_sigma_closure.py, run 2026-07-10 on NERSC).")}
    for name, fn in VARIANTS.items():
        d = json.loads((SRC / fn).read_text())
        bins = []
        for b in d["bins"]:
            entry = {"sample": b["sample"], "zmin": b["zmin"],
                     "zmax": b["zmax"],
                     "logms_med": round(b["logms_med"], 3),
                     "logmh_med": round(b.get("logmh_med", float("nan")), 3),
                     "A_1h": {}}
            for sv, e in b["surveys"].items():
                w = e["windows"].get("one_halo")
                if w:
                    entry["A_1h"][sv] = [round(w["A_ds"], 3),
                                         round(w["A_ds_err"], 3)]
            bins.append(entry)
        out[name] = bins
    OUT.write_text(json.dumps(out, indent=1))
    for name in VARIANTS:
        print(f"--- {name}")
        for b in out[name]:
            vals = " ".join(f"{sv}:{a[0]:.2f}±{a[1]:.2f}"
                            for sv, a in b["A_1h"].items())
            print(f"  {b['sample']:10s} z[{b['zmin']},{b['zmax']}] "
                  f"logM*={b['logms_med']:.2f} | {vals}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
