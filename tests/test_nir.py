"""NirDirect estimator + snkappa.nir join (deep-NIR sprint P1)."""

import numpy as np
import pandas as pd
import pytest
from astropy.cosmology import FlatLambdaCDM

from snkappa.stellar import Nir1um, NirDirect, NirDirectFSF, Nir1umFSF

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def _mags(n, **kw):
    base = {b: np.full(n, np.nan) for b in
            ("g", "r", "i", "z", "y", "j", "h", "ks", "w1")}
    base.update({k: np.asarray(v, dtype=float) for k, v in kw.items()})
    return base


def test_band_center_exact():
    """At (1+z) = lambda_H the estimate uses the H mag exactly."""
    est = NirDirect(COSMO)
    z = np.array([0.65])                       # (1+z) = 1.65 um = H
    mags = _mags(1, z=[20.0], j=[19.0], h=[18.5], ks=[18.2], w1=[18.0])
    m_t, used = est._interp_1um(mags, z)
    assert used[0]
    assert m_t[0] == pytest.approx(18.5, abs=1e-9)


def test_fallback_equals_nir1um():
    """No deep-NIR photometry -> exactly the Nir1um z <-> W1 result."""
    rng = np.random.default_rng(7)
    n = 200
    z = rng.uniform(0.05, 1.1, n)
    mags = _mags(n, z=rng.uniform(18, 22.5, n), w1=rng.uniform(17, 21, n),
                 g=rng.uniform(19, 23, n))
    old = Nir1um(COSMO).logmstar(mags, z)
    new, used = NirDirect(COSMO).logmstar_flagged(mags, z)
    assert not used.any()
    np.testing.assert_allclose(new, old, rtol=0, atol=1e-12)


def test_powerlaw_sed_invariance():
    """If the SED is a pure power law, every bracketing choice agrees."""
    est = NirDirect(COSMO)
    n = 50
    z = np.linspace(0.05, 1.1, n)
    alpha = -1.3
    lam = dict(est.LADDER)
    mags = _mags(n, **{b: 20.0 - 2.5 * alpha * np.log10(lam[b] / 0.92)
                       for b in ("z", "y", "j", "h", "ks", "w1")})
    new = est.logmstar(mags, z)
    mags_w1 = _mags(n, z=mags["z"], w1=mags["w1"])
    old = Nir1um(COSMO).logmstar(mags_w1, z)
    np.testing.assert_allclose(new, old, atol=1e-9)


def test_bump_sed_diverges():
    """A bump-distorted SED gives different (better) short-baseline masses:
    the z <-> W1 straight line misses the H-band excess."""
    est = NirDirect(COSMO)
    z = np.array([0.65])
    # H brighter than the z<->W1 power law would predict
    mags = _mags(1, z=[20.0], h=[18.0], w1=[19.0])
    m_t, used = est._interp_1um(mags, z)
    assert used[0]
    assert m_t[0] == pytest.approx(18.0)     # sits exactly on H
    m_t_old = Nir1um(COSMO)                  # z<->W1 interp at 1.65 um
    frac = np.log(1.65 / 0.92) / np.log(3.4 / 0.92)
    assert abs(m_t[0] - (20.0 + frac * (19.0 - 20.0))) > 0.5


def test_missing_w1_with_nir():
    """Ks-only bracket: (1+z) beyond Ks but no W1 -> no upper band, falls
    back to Taylor (needs g); with W1 present the ks <-> w1 pair is used."""
    est = NirDirect(COSMO)
    z = np.array([1.4])                       # lambda_t = 2.4 um > Ks
    mags = _mags(1, z=[21.0], ks=[19.0], g=[22.0])
    m_t, used = est._interp_1um(mags, z)
    assert not np.isfinite(m_t[0]) and not used[0]
    mags = _mags(1, z=[21.0], ks=[19.0], w1=[18.8])
    m_t, used = est._interp_1um(mags, z)
    assert np.isfinite(m_t[0]) and used[0]


def test_fsf_fallback_matches_nir1um_fsf():
    """Outside the NIR footprint NirDirectFSF == Nir1umFSF exactly."""
    rng = np.random.default_rng(11)
    n = 100
    z = rng.uniform(0.05, 1.1, n)
    mags = _mags(n, z=rng.uniform(18, 22.5, n), w1=rng.uniform(17, 21, n))
    old = Nir1umFSF(COSMO).logmstar(mags, z)
    new, used = NirDirectFSF(COSMO).logmstar_flagged(mags, z)
    assert not used.any()
    np.testing.assert_allclose(new, old, atol=1e-12)


def test_attach_nir(tmp_path):
    from snkappa.nir import attach_nir
    nir = pd.DataFrame({
        "ra": [10.0, 10.001, 10.5], "dec": [-44.0, -44.001, -44.2],
        "y_mag_auto": [20.0, 21.0, 19.5], "y_magerr_auto": [.02, .03, .02],
        "j_mag_auto": [19.5, 20.5, 19.0], "j_magerr_auto": [.02, .03, .02],
        "h_mag_auto": [19.0, 20.0, 18.5], "h_magerr_auto": [.02, .03, .02],
        "ks_mag_auto": [18.8, 19.8, 18.2],
        "ks_magerr_auto": [.02, .03, .02],
        "haloflag": [0, 0, 1]})
    nir.to_parquet(tmp_path / "video_es1.parquet")
    df = pd.DataFrame({
        "ra": [10.0, 12.0], "dec": [-44.0, -44.0],
        "mw_transmission_z": [1.0, 1.0]})
    out = attach_nir(df, data_dir=tmp_path)
    assert out.attrs["n_nir_matched"] == 1
    assert out["mag_j"].iloc[0] == pytest.approx(19.5)
    assert np.isnan(out["mag_j"].iloc[1])
    # haloflag source is dropped even if it were the nearest
    df2 = pd.DataFrame({"ra": [10.5], "dec": [-44.2],
                        "mw_transmission_z": [1.0]})
    out2 = attach_nir(df2, data_dir=tmp_path)
    assert out2.attrs["n_nir_matched"] == 0


def test_attach_nir_extinction(tmp_path):
    from snkappa.nir import attach_nir, R_NIR, R_LS_Z
    nir = pd.DataFrame({
        "ra": [10.0], "dec": [-44.0],
        "y_mag_auto": [20.0], "y_magerr_auto": [.02],
        "j_mag_auto": [19.5], "j_magerr_auto": [.02],
        "h_mag_auto": [19.0], "h_magerr_auto": [.02],
        "ks_mag_auto": [18.8], "ks_magerr_auto": [.02]})
    nir.to_parquet(tmp_path / "video_es1.parquet")
    mwt = 0.97          # A_z ~ 0.033 mag -> E(B-V) ~ 0.027
    df = pd.DataFrame({"ra": [10.0], "dec": [-44.0],
                       "mw_transmission_z": [mwt]})
    out = attach_nir(df, data_dir=tmp_path)
    ebv = -2.5 * np.log10(mwt) / R_LS_Z
    assert out["mag_j"].iloc[0] == pytest.approx(19.5 - R_NIR["j"] * ebv)
