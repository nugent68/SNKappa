"""Surface-density profile checks: analytic NFW vs numerical integration,
BMO spline accuracy, truncation behavior, and a point-mass sanity limit."""

import numpy as np

from snkappa.halos import (bmo_sigma_direct, bmo_table, nfw_sigma_dimless)


def test_nfw_analytic_vs_numerical():
    """W&B analytic NFW == direct integration with negligible truncation."""
    x = np.array([0.01, 0.1, 0.5, 0.99, 1.0, 1.01, 2.0, 10.0, 50.0])
    numerical = bmo_sigma_direct(x, tau=1e4, n_quad=400)
    analytic = nfw_sigma_dimless(x)
    np.testing.assert_allclose(numerical, analytic, rtol=2e-3)


def test_bmo_table_accuracy():
    rng = np.random.default_rng(7)
    x = 10 ** rng.uniform(-3.5, 2.5, 60)
    for tau in (2.0, 7.0, 25.0, 120.0):
        table_val = bmo_table().sigma_dimless(x, np.full_like(x, tau))
        direct = bmo_sigma_direct(x, tau, n_quad=400)
        np.testing.assert_allclose(table_val, direct, rtol=0.02)


def test_truncation_suppresses_outskirts():
    x = np.array([0.1, 1.0, 5.0, 20.0])
    trunc = bmo_table().sigma_dimless(x, np.full_like(x, 5.0))
    full = nfw_sigma_dimless(x)
    assert (trunc < full).all()
    # inside r_s truncation is a mild effect; far outside it dominates
    assert trunc[0] / full[0] > 0.7
    assert trunc[-1] / full[-1] < 0.3


def test_sis_pointmass_kappa_scalings():
    """Aggregate kappa scalings: NFW Sigma ~ 1/x^2 * ln at large x (steeper
    than SIS 1/x), and mass inside aperture is finite for BMO."""
    x = np.logspace(0.5, 2.5, 40)
    s = nfw_sigma_dimless(x)
    slope = np.gradient(np.log(s), np.log(x))
    assert (slope < -1.5).all()  # steeper than SIS everywhere out here
    # BMO enclosed 2D mass converges: negligible mass beyond x ~ 20*tau
    xg = np.logspace(-3, 2.9, 400)
    integrand = xg * bmo_table().sigma_dimless(xg, np.full_like(xg, 5.0))
    cum = np.cumsum(integrand[:-1] * np.diff(xg))
    i100 = np.searchsorted(xg, 100.0)
    assert cum[-1] / cum[i100] < 1.02


def test_halo_amplitude_cylinder_mass():
    """End-to-end amplitude check of the (M200,c)->(rho_s,r_s) chain: the
    projected (cylinder) mass of the truncated halo within r200 must be
    within ~30% of M200 -- catches any unit error in rho_s*r_s."""
    from astropy.cosmology import FlatLambdaCDM
    from snkappa.config import HaloModelConfig
    from snkappa.halos import HaloModel

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    hm = HaloModel(HaloModelConfig(), cosmo, z_src=2.0, dz=0.05)
    ib = hm.zbin_index([0.4])
    # logM* chosen so the SMHM gives ~10^13 Msun (uncapped regime)
    rhos, rs, tau = hm.halo_params(np.array([11.0]), ib)
    m200 = 4.0 / 3.0 * np.pi * 200.0 * hm.rhoc[ib[0]] * (rs * tau) ** 3
    x = np.linspace(1e-3, tau[0], 3000)
    sigma = rhos[0] * rs[0] * hm.sigma_dimless(x, np.full_like(x, tau[0]))
    m_cyl = np.trapezoid(sigma * 2 * np.pi * (x * rs[0]), x * rs[0])
    assert 0.7 < m_cyl / m200[0] < 1.3
