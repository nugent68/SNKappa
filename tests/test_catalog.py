"""Catalog hygiene: Data Lab NULL sentinels must never fabricate photometry."""

import numpy as np
import pandas as pd

from snkappa.catalog import dered_mag


def test_sentinel_flux_gives_nan_not_mag225():
    flux = pd.Series([-9999.0, 10.0, 0.0, -1.0])
    mw = pd.Series([-9999.0, 0.9, 0.8, 0.9])
    mag = dered_mag(flux, mw)
    assert np.isnan(mag.iloc[0])          # the (-9999)/(-9999)=+1 trap
    assert abs(mag.iloc[1] - (22.5 - 2.5 * np.log10(10.0 / 0.9))) < 1e-6
    assert np.isnan(mag.iloc[2]) and np.isnan(mag.iloc[3])


def test_sentinel_transmission_gives_nan():
    flux = pd.Series([10.0, 10.0])
    mw = pd.Series([0.0, 5.0])            # zero and unphysical > 1
    mag = dered_mag(flux, mw)
    assert mag.isna().all()
