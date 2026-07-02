"""Engine-level tests on synthetic catalogs: deflector exclusion, null field,
and reproducibility."""

import numpy as np

from snkappa.kappa import KappaEngine
from snkappa.stellar import make_estimator

from conftest import synthetic_catalog


def _engine(cfg, halo_model, df):
    est = make_estimator(cfg.halo_model.mstar_method, cfg.cosmo)
    return KappaEngine(cfg, cfg.cosmo, halo_model, est, df)


def test_exclusion_removes_expected_kappa(cfg, halo_model):
    rng = np.random.default_rng(3)
    df = synthetic_catalog(rng, 400, cfg.source.ra_src, cfg.source.dec_src,
                           0.2, cfg.source.z_src)
    # flag the 5 nearest galaxies as "deflector/group"
    from snkappa.kappa import angular_sep_arcsec
    theta = angular_sep_arcsec(cfg.source.ra_src, cfg.source.dec_src,
                               df["ra"].to_numpy(), df["dec"].to_numpy())
    nearest = np.argsort(theta)[:5]
    excl = np.zeros(len(df), dtype=bool)
    excl[nearest] = True

    eng = _engine(cfg, halo_model, df)
    r_ap = cfg.los.aperture_radius_arcmin * 60.0
    idx_all, _, k_all = eng.kappa_los(cfg.source.ra_src, cfg.source.dec_src,
                                      0.0, r_ap)
    idx_cut, _, k_cut = eng.kappa_los(cfg.source.ra_src, cfg.source.dec_src,
                                      0.0, r_ap, exclude_mask=excl)
    removed = k_all.sum() - k_cut.sum()
    expected = k_all[np.isin(idx_all, nearest)].sum()
    assert expected > 0
    np.testing.assert_allclose(removed, expected, rtol=1e-10)
    assert not np.isin(idx_cut, nearest).any()


def test_null_field_kappa_ext_consistent_with_zero(cfg, halo_model):
    """Uniform synthetic field: SN-position kappa minus the random-LOS mean
    must scatter around zero."""
    rng = np.random.default_rng(5)
    df = synthetic_catalog(rng, 4000, cfg.source.ra_src, cfg.source.dec_src,
                           1.0, cfg.source.z_src)
    eng = _engine(cfg, halo_model, df)
    r_ap = cfg.los.aperture_radius_arcmin * 60.0
    # central exclusion as in the real pipeline: without it the log-divergent
    # halo centers make the kappa sum nearest-galaxy dominated (heavy tails)
    r_in = 30.0

    k_sn = eng.kappa_los(cfg.source.ra_src, cfg.source.dec_src,
                         r_in, r_ap)[2].sum()
    k_rand = []
    for _ in range(60):
        r = 0.5 * np.sqrt(rng.uniform(0, 1))
        phi = rng.uniform(0, 2 * np.pi)
        ra0 = cfg.source.ra_src + r * np.cos(phi)
        dec0 = cfg.source.dec_src + r * np.sin(phi)
        k_rand.append(eng.kappa_los(ra0, dec0, r_in, r_ap)[2].sum())
    k_rand = np.array(k_rand)
    sig = k_rand.std(ddof=1)
    assert sig > 0
    assert abs(k_sn - k_rand.mean()) < 4.0 * sig


def test_reproducibility_same_seed(cfg, halo_model):
    from snkappa import montecarlo
    rng1 = np.random.default_rng(99)
    df = synthetic_catalog(rng1, 500, cfg.source.ra_src, cfg.source.dec_src,
                           0.2, cfg.source.z_src)
    eng = _engine(cfg, halo_model, df)
    excl = np.zeros(len(df), dtype=bool)
    d1 = montecarlo.mc_kappa_raw(cfg, eng, np.random.default_rng(42), excl)
    d2 = montecarlo.mc_kappa_raw(cfg, eng, np.random.default_rng(42), excl)
    np.testing.assert_array_equal(d1, d2)
    assert d1.std() > 0


def test_monster_mstar_capped_halo(cfg, halo_model):
    """A photometric-junk galaxy (clipped logM*=12.2) must not become a
    cluster: halo mass capped at logmh_max."""
    rhos, rs, tau = halo_model.halo_params(np.array([12.2]), np.array([10]))
    rhoc = halo_model.rhoc[10]
    m200 = 4.0 / 3.0 * np.pi * 200.0 * rhoc * (rs * tau) ** 3
    assert np.log10(m200[0]) <= cfg.halo_model.logmh_max + 1e-6
