"""Nir1umFSF recalibration: sane, monotone, small at low mass."""

import numpy as np
from astropy.cosmology import Planck18

from snkappa.stellar import make_estimator


def _mags(m_z, m_w1):
    n = len(m_z)
    return {"g": np.full(n, 21.0), "z": np.asarray(m_z, float),
            "w1": np.asarray(m_w1, float)}


def test_recal_small_at_low_mass_positive_at_high():
    e0 = make_estimator("nir1um", Planck18)
    e1 = make_estimator("nir1um_fsf", Planck18)
    mags = _mags([20.5, 17.0, 15.5], [20.3, 16.8, 15.0])
    z = np.array([0.2, 0.4, 0.3])
    raw = e0.logmstar(mags, z)
    cor = e1.logmstar(mags, z)
    d = cor - raw
    assert abs(d[0]) < 0.12          # low mass: no big shift
    assert 0.15 < d[1] < 0.55        # massive red: substantial positive
    assert 0.15 < d[2] < 0.55
    # correction declines with redshift at fixed (massive) galaxy
    mags1 = _mags([17.0], [16.8])
    d_lo = (e1.logmstar(mags1, np.array([0.3]))
            - e0.logmstar(mags1, np.array([0.3])))[0]
    d_hi = (e1.logmstar(mags1, np.array([1.0]))
            - e0.logmstar(mags1, np.array([1.0])))[0]
    assert d_hi < d_lo


def test_recal_monotone_in_raw_mass():
    e1 = make_estimator("nir1um_fsf", Planck18)
    zz = np.full(30, 0.35)
    m_z = np.linspace(21.5, 15.0, 30)      # brighter -> more massive
    out = e1.logmstar(_mags(m_z, m_z - 0.3), zz)
    assert np.all(np.diff(out) >= -1e-6)
