"""Corrections delivered to the strong-lens model.

Mass-sheet transform with external sheet kappa_ext (Falco et al. 1985):
  magnification:  mu_true = mu_model / (1 - kappa_ext)^2
                  -> multiplicative flux correction (1 - kappa_ext)^2,
                     magnitude offset  Delta_m = +2.5 log10[(1-kappa_ext)^2]
  time delays:    H0_true = (1 - kappa_ext) * H0_model

SINGLE-PLANE external-convergence approximation: all LOS halos are compressed
into one effective sheet at the main-lens plane. A rigorous treatment is
multi-plane ray tracing (e.g. lenstronomy) using the per-galaxy catalog this
pipeline writes; quote P(kappa_ext) with that caveat.
"""

from __future__ import annotations

import numpy as np

from .montecarlo import percentiles

CAVEAT = ("kappa_ext is a single-plane external-convergence (mass-sheet) "
          "approximation; for a rigorous treatment use multi-plane ray "
          "tracing with the per-galaxy LOS catalog written by this pipeline.")


def corrections_summary(kappa_samples: np.ndarray) -> dict:
    one_minus = 1.0 - kappa_samples
    flux_corr = one_minus**2
    dm = 2.5 * np.log10(np.clip(flux_corr, 1e-6, None))
    return {
        "kappa_ext": percentiles(kappa_samples),
        "flux_correction_(1-k)^2": percentiles(flux_corr),
        "delta_m_mag": percentiles(dm),
        "H0_scaling_(1-k)": percentiles(one_minus),
        "convention": CAVEAT,
    }
