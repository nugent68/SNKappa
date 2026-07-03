"""Cluster-halo tier: profile amplitude, host/field split, member
replacement bookkeeping, and miscentering Monte Carlo."""

import numpy as np
import pandas as pd
import pytest

from snkappa import clusters as clu
from snkappa.halos import HaloModel
from snkappa.kappa import sigma_crit_msun_mpc2

from conftest import synthetic_catalog


@pytest.fixture()
def cl_df(cfg):
    return pd.DataFrame([
        {"name": "host", "ra": cfg.deflector.ra_lens,
         "dec": cfg.deflector.dec_lens, "z": 0.4, "m200": 2e14,
         "source": "manual"},
        {"name": "field", "ra": cfg.source.ra_src + 0.1,
         "dec": cfg.source.dec_src, "z": 0.5, "m200": 1e14,
         "source": "manual"},
    ])


def test_split_host(cfg, cl_df):
    hosts, field = clu.split_host(cfg, cl_df)
    assert list(hosts["name"]) == ["host"]
    assert list(field["name"]) == ["field"]


def test_cluster_kappa_amplitude(cfg, halo_model, cl_df):
    """kappa from ClusterKappa matches an independent NFW evaluation."""
    ck = clu.ClusterKappa(cfg, halo_model, cfg.cosmo, cl_df.iloc[[0]])
    theta = 120.0  # arcsec offset
    dec0 = cfg.deflector.dec_lens + theta / 3600.0
    k = ck.kappa_sum(cfg.deflector.ra_lens, dec0)

    ib = halo_model.zbin_index([0.4])[0]
    rhoc = halo_model.rhoc[ib]
    c = cfg.clusters.concentration
    r200 = (3 * 2e14 / (4 * np.pi * 200 * rhoc)) ** (1 / 3)
    rs = r200 / c
    rhos = 200 / 3 * rhoc * c**3 / (np.log(1 + c) - c / (1 + c))
    scr = sigma_crit_msun_mpc2(cfg.cosmo, [halo_model.zbins[ib]],
                               cfg.source.z_src)[0]
    x = (theta / 206264.806) * halo_model.da[ib] / rs
    expected = rhos * rs * halo_model.sigma_dimless(
        np.array([x]), np.array([c]))[0] / scr
    np.testing.assert_allclose(k, expected, rtol=1e-6)
    assert 0.001 < k < 0.5  # sane cluster-scale convergence


def test_member_replacement_bookkeeping(cfg, halo_model, cl_df):
    """Members inside theta_200 and dz are flagged; excluding them removes
    exactly their galaxy-halo kappa from the sum."""
    from snkappa.kappa import KappaEngine
    from snkappa.stellar import make_estimator

    rng = np.random.default_rng(17)
    df = synthetic_catalog(rng, 500, cfg.source.ra_src, cfg.source.dec_src,
                           0.2, cfg.source.z_src)
    # plant guaranteed members: spec-z at the host cluster redshift, close in
    df.loc[:9, "z_spec"] = 0.4 + rng.normal(0, 0.001, 10)
    df.loc[:9, "ra"] = cfg.source.ra_src + rng.uniform(-0.01, 0.01, 10)
    df.loc[:9, "dec"] = cfg.source.dec_src + rng.uniform(-0.01, 0.01, 10)

    members = clu.assign_members(cfg, df, cl_df, halo_model)
    assert members[:10].all()

    est = make_estimator(cfg.halo_model.mstar_method, cfg.cosmo)
    eng = KappaEngine(cfg, cfg.cosmo, halo_model, est, df)
    r_ap = cfg.los.aperture_radius_arcmin * 60.0
    i_all, _, k_all = eng.kappa_los(cfg.source.ra_src, cfg.source.dec_src,
                                    1.0, r_ap)
    i_cut, _, k_cut = eng.kappa_los(cfg.source.ra_src, cfg.source.dec_src,
                                    1.0, r_ap, exclude_mask=members)
    removed = k_all[np.isin(i_all, np.flatnonzero(members))].sum()
    assert removed > 0
    np.testing.assert_allclose(k_all.sum() - k_cut.sum(), removed,
                               rtol=1e-10)


def test_miscentering_mc(cfg, halo_model, cl_df):
    """Miscentering draws broaden and (for a centered sightline) lower the
    cluster kappa relative to the perfectly centered value."""
    ck = clu.ClusterKappa(cfg, halo_model, cfg.cosmo, cl_df.iloc[[0]])
    ra0, dec0 = cfg.deflector.ra_lens, cfg.deflector.dec_lens
    k_cen = ck.kappa_sum(ra0, dec0)
    rng = np.random.default_rng(3)
    draws = ck.mc_kappa_sum(ra0, dec0, rng, 300)
    assert draws.std() > 0
    assert np.median(draws) < k_cen  # off-center halo -> lower central kappa


def test_sigma_v_diagnostic(cfg):
    rng = np.random.default_rng(8)
    n = 40
    df = pd.DataFrame({
        "ra": cfg.source.ra_src + rng.normal(0, 0.02, n),
        "dec": cfg.source.dec_src + rng.normal(0, 0.02, n),
        "z_spec": 0.4 + rng.normal(0, 600 / 299792.458, n) * 1.4,
    })
    out = clu.velocity_dispersion(cfg, df, cfg.source.ra_src,
                                  cfg.source.dec_src, 0.4, rmax_arcmin=5)
    assert out["n_members"] > 20
    assert 350 < out["sigma_v_kms"] < 900
    assert 1e13 < out["m200_dyn"] < 1e15
