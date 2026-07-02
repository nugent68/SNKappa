"""Sigma_crit validated against the classic SIS Einstein-radius formula.

For an SIS with velocity dispersion sigma_v:
    theta_E = 4 pi (sigma_v/c)^2 D_ls/D_s      (independent, textbook)
    Sigma_SIS(R) = sigma_v^2 / (2 G R)
    kappa(theta) = Sigma(theta D_l) / Sigma_crit  must equal  theta_E/(2 theta).
Any error in Sigma_crit units or distance combination breaks this equality.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from snkappa.kappa import sigma_crit_msun_mpc2


def test_sigma_crit_sis_einstein_radius():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    zl, zs = 0.4, 2.0
    sigma_v = 250 * u.km / u.s

    d_l = cosmo.angular_diameter_distance(zl)
    d_s = cosmo.angular_diameter_distance(zs)
    d_ls = cosmo.angular_diameter_distance_z1z2(zl, zs)
    theta_e = (4 * np.pi * (sigma_v / const.c) ** 2 * d_ls / d_s).to(
        u.dimensionless_unscaled).value  # radians

    scr = sigma_crit_msun_mpc2(cosmo, zl, zs)[0]  # Msun/Mpc^2
    theta = np.logspace(-6, -4, 10)  # radians
    r_phys = theta * d_l.to(u.Mpc).value  # Mpc
    sigma_sis = (sigma_v**2 / (2 * const.G)).to(u.Msun / u.Mpc).value / r_phys
    kappa = sigma_sis / scr

    np.testing.assert_allclose(kappa, theta_e / (2 * theta), rtol=1e-6)


def test_sigma_crit_limits():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    zl = np.array([0.001, 0.5, 1.9999])
    scr = sigma_crit_msun_mpc2(cosmo, zl, 2.0)
    assert scr[0] > scr[1] and scr[2] > scr[1]  # diverges at both ends
    assert np.isinf(sigma_crit_msun_mpc2(cosmo, [2.0], 2.0)[0])
