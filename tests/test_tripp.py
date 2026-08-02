"""Tripp standardization: synthetic recovery of (alpha, beta, gamma)
and per-sample sigma_int."""

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM


def test_tripp_recovers_parameters():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "scripts"))
    from union3_prep import tripp_fit, OM_FID

    rng = np.random.default_rng(42)
    n = 1500
    z = rng.uniform(0.05, 1.0, n)
    x1 = rng.normal(0, 1, n)
    c = rng.normal(0, 0.08, n)
    mass = rng.uniform(8.5, 11.5, n)
    samples = rng.choice(["A", "B"], n)
    a_t, b_t, M_t, g_t = 0.15, 3.0, -19.3, 0.06
    s_int = np.where(samples == "A", 0.08, 0.16)
    cosmo = FlatLambdaCDM(H0=70, Om0=OM_FID)
    sig_m = 0.05
    mB = (cosmo.distmod(z).value - a_t * x1 + b_t * c + M_t
          + g_t * (mass > 10.0)
          + rng.standard_normal(n) * np.hypot(s_int, sig_m))
    cov = np.zeros((n, 3, 3))
    cov[:, 0, 0] = sig_m**2
    cov[:, 1, 1] = 1e-4
    cov[:, 2, 2] = 1e-6

    df = pd.DataFrame({"zHD": z, "x1": x1, "c": c, "mB": mB,
                       "HOST_LOGMASS": mass, "sample": samples})
    params, mu, muerr, pull = tripp_fit(df, cov, np.ones(n, dtype=bool))
    assert abs(params["alpha"] - a_t) < 0.02
    assert abs(params["beta"] - b_t) < 0.15
    assert abs(params["gamma_mass_step"] - g_t) < 0.03
    assert abs(params["sig_int_by_sample"]["A"] - 0.08) < 0.03
    assert abs(params["sig_int_by_sample"]["B"] - 0.16) < 0.03
    # residuals vs LCDM are centered and pulls ~ unit
    hr = mu - cosmo.distmod(z).value
    assert abs(np.mean(hr)) < 0.01
    assert 0.85 < np.std(pull) < 1.15
