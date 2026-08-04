#!/usr/bin/env python
"""8-band FrankenBlast over the SNKappa top-kappa contributor catalog.

Thin wrapper around run_lens_catalog.py: same SBI model (zfree_GPD2W),
same fitting/postprocessing path, but the band map carries the deep-NIR
photometry packaged by SNKappa scripts/fb_targets.py — DECam g/r/z +
VISTA J/H/Ks (served through the 2MASS filter curves the SBI model was
trained on; VISTA-2MASS color terms are well below the photometric
errors) + WISE W1/W2. The input CSV pre-dereddens the NIR fluxes, so
their mw_transmission columns are 1.0.

Env overrides (OOD-churn mitigation for very bright targets; the SBI++
missing/noisy-band loop retries without bound when the conditioning
vector sits outside the training distribution):
  KFB_TMAX_PER_ITER  seconds per sampling iteration (default: the
                     campaign quick-look value 20; SBI++ default is 60)
  KFB_ERR_FLOOR      fractional flux-error floor, e.g. 0.05 — justified
                     by the ~0.1 mag aperture systematics of extended
                     bright galaxies (LS total vs VIDEO AUTO)

Usage (identical to run_lens_catalog):
    python run_kappa_catalog.py <csv> <start> <end> <summary_csv>
"""

import os

import numpy as np

import run_lens_catalog as base
import fit_host_sed

base.BANDMAP = [
    ("DES_g", "flux_g", "flux_ivar_g", "mw_transmission_g"),
    ("DES_r", "flux_r", "flux_ivar_r", "mw_transmission_r"),
    ("DES_z", "flux_z", "flux_ivar_z", "mw_transmission_z"),
    ("2MASS_J", "flux_j", "flux_ivar_j", "mw_transmission_j"),
    ("2MASS_H", "flux_h", "flux_ivar_h", "mw_transmission_h"),
    ("2MASS_K", "flux_k", "flux_ivar_k", "mw_transmission_k"),
    ("WISE_W1", "flux_w1", "flux_ivar_w1", "mw_transmission_w1"),
    ("WISE_W2", "flux_w2", "flux_ivar_w2", "mw_transmission_w2"),
]

_tmx = os.environ.get("KFB_TMAX_PER_ITER")
if _tmx:
    fit_host_sed.run_params["tmax_per_iter"] = int(_tmx)

_npost = os.environ.get("KFB_NPOSTERIOR")
if _npost:
    fit_host_sed.run_params["nposterior"] = int(_npost)

_floor = float(os.environ.get("KFB_ERR_FLOOR", "0") or 0)
if _floor > 0:
    _orig_build = base.build_observations

    def _build_with_floor(row):
        obs = _orig_build(row)
        obs["maggies_unc"] = np.maximum(obs["maggies_unc"],
                                        _floor * obs["maggies"])
        return obs

    base.build_observations = _build_with_floor

if __name__ == "__main__":
    base.main()
