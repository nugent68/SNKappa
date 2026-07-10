"""BMO Delta Sigma table vs the analytic NFW excess surface density.

At tau = 300 (truncation far outside the radii probed) the BMO profile is
an NFW, so DeltaSigma/(rho_s r_s) must match the closed-form
Wright & Brainerd (2000) expression.
"""

import numpy as np

from snkappa.halos import bmo_table


def nfw_delta_sigma_dimless(x):
    """Wright & Brainerd 2000: (Sigma_bar(<x) - Sigma(x)) / (rho_s r_s)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi < 1.0 - 1e-9:
            s = np.arctanh(np.sqrt((1 - xi) / (1 + xi)))
            g = (8 * s / (xi**2 * np.sqrt(1 - xi**2))
                 + 4 / xi**2 * np.log(xi / 2)
                 - 2 / (xi**2 - 1)
                 + 4 * s / ((xi**2 - 1) * np.sqrt(1 - xi**2)))
        elif xi > 1.0 + 1e-9:
            s = np.arctan(np.sqrt((xi - 1) / (1 + xi)))
            g = (8 * s / (xi**2 * np.sqrt(xi**2 - 1))
                 + 4 / xi**2 * np.log(xi / 2)
                 - 2 / (xi**2 - 1)
                 + 4 * s / (xi**2 - 1) ** 1.5)
        else:
            g = 10.0 / 3.0 + 4.0 * np.log(0.5)
        out[i] = g
    return out


def test_bmo_delta_sigma_matches_nfw():
    x = np.logspace(-1.5, 0.7, 25)
    ds_tab = bmo_table().delta_sigma_dimless(x, np.full_like(x, 300.0))
    ds_ana = nfw_delta_sigma_dimless(x)
    assert np.all(np.abs(ds_tab / ds_ana - 1.0) < 0.02)


def test_delta_sigma_positive_and_declining():
    x = np.logspace(-1, 1, 30)
    ds = bmo_table().delta_sigma_dimless(x, np.full_like(x, 5.0))
    assert (ds > 0).all()
    assert ds[0] > ds[-1]
