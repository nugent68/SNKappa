#!/usr/bin/env python
"""8-band FrankenBlast with the redshift FIXED from the SNKappa catalog.

The campaign runner always sets obs["redshift"] = None, which forces the
z-free SBI model (sbi_pp_photoz) to infer redshift along with everything
else. But SNKappa already knows every galaxy's redshift -- DESI spec-z
where available, otherwise the Legacy Survey (DECaLS) photo-z that the
kappa pipeline itself uses -- and the campaign ships a redshift-
CONDITIONED model, SBI_model_zfix_GPD2W_global.pt, whose conditioning
vector is [fluxes, uncertainties, redshift] (sbi_pp.py line 128).

Fixing z should (a) put the mass on the right distance -- the z-free
fits missed a known spec-z by 0.15 in z and drifted 5.4% in (1+z)
median -- and (b) shrink the inference problem, which is the plausible
cure for the OOD churn that times out ~80% of the bright targets.

Usage: python run_kappa_zfix.py <csv> <start> <end> <summary_csv>
"""

import os

import numpy as np

import run_kappa_catalog  # installs the 8-band BANDMAP + env knobs
import run_lens_catalog as base

SBIPP_ROOT = os.environ["SBIPP_ROOT"]
SBIPP_PHOT_ROOT = os.environ["SBIPP_PHOT_ROOT"]

base.SBI_PARAMS = {
    "anpe_fname_global": f"{SBIPP_ROOT}/SBI_model_zfix_GPD2W_global.pt",
    "train_fname_global":
        f"{SBIPP_PHOT_ROOT}/sbi_phot_zfix_GPD2W_global.h5",
    "nhidden": 500,
    "nblocks": 15,
}
base.TRAIN_FNAME = "zfix_GPD2W"

_orig_build = base.build_observations


def _build_with_z(row):
    obs = _orig_build(row)
    z = row.get("z_best", np.nan)
    if not np.isfinite(z):
        raise RuntimeError("no redshift available for zfix mode")
    obs["redshift"] = float(z)
    return obs


base.build_observations = _build_with_z

if __name__ == "__main__":
    base.main()
