"""Shear tier: tangential amplitude wiring, spin-2 vector addition, and
the cluster-field DeltaSigma profile."""

import numpy as np
import pandas as pd
import pytest

from snkappa.batch import BatchEngine, ClusterField

ZC = np.arange(0.02, 1.9, 0.1)


def _one_gal_df(ra, dec, z_spec):
    return pd.DataFrame([{
        "ra": ra, "dec": dec, "z_spec": z_spec,
        "zp_med": np.nan, "zp_std": np.nan,
        "zp_l68": np.nan, "zp_u68": np.nan,
        "mag_g": 20.4, "mag_r": 19.7, "mag_i": 19.3, "mag_z": 19.0,
    }])


@pytest.fixture()
def est(cfg):
    from snkappa.stellar import make_estimator
    return make_estimator(cfg.halo_model.mstar_method, cfg.cosmo)


def test_single_halo_shear_amplitude_and_orientation(cfg, halo_model, est):
    """Halo due north: gamma_t = amp * DeltaSigma_dimless, purely gamma1>0
    (tangential stretch is east-west), gamma2 = 0."""
    ra0, dec0 = 150.0, 30.0
    theta = 100.0  # arcsec
    df = _one_gal_df(ra0, dec0 + theta / 3600.0, 0.4)
    eng = BatchEngine(cfg, df, halo_model, est, ZC, r_in_arcsec=3.0)
    eng.set_zsrc(cfg.cosmo, 1.0)

    k, g1, g2 = eng.kappa_shear_gal(ra0, dec0, 600.0)
    assert k == pytest.approx(eng.kappa_gal(ra0, dec0, 600.0), rel=1e-12)

    x = np.array([theta / eng.s_ths[0]])
    tau = np.array([eng.s_tau[0]])
    gt_exp = float(eng.sA[0] * halo_model.delta_sigma_dimless(x, tau)[0])
    assert gt_exp > 0
    assert g1 == pytest.approx(gt_exp, rel=1e-6)
    assert abs(g2) < 1e-12 * max(gt_exp, 1e-30)
    # for these scales gamma_t > kappa (steeply falling profile)
    assert g1 > k


def test_two_halo_shear_vector_cancellation(cfg, halo_model, est):
    """Identical halos due N and due E at the same radius: kappa adds
    scalarly (doubles) but their gamma1 contributions cancel exactly --
    the spin-2 regression test."""
    ra0, dec0 = 150.0, 30.0
    theta = 100.0
    df = pd.concat([
        _one_gal_df(ra0, dec0 + theta / 3600.0, 0.4),                 # N
        _one_gal_df(ra0 + theta / 3600.0 / np.cos(np.radians(dec0)),
                    dec0, 0.4),                                        # E
    ], ignore_index=True)
    eng = BatchEngine(cfg, df, halo_model, est, ZC, r_in_arcsec=3.0)
    eng.set_zsrc(cfg.cosmo, 1.0)
    k2, g1, g2 = eng.kappa_shear_gal(ra0, dec0, 600.0)

    df1 = _one_gal_df(ra0, dec0 + theta / 3600.0, 0.4)
    eng1 = BatchEngine(cfg, df1, halo_model, est, ZC, r_in_arcsec=3.0)
    eng1.set_zsrc(cfg.cosmo, 1.0)
    k1, g1_single, _ = eng1.kappa_shear_gal(ra0, dec0, 600.0)

    assert k2 == pytest.approx(2.0 * k1, rel=1e-3)   # haversine E-offset
    # N contributes +gt to gamma1, E contributes -gt: near-total cancel
    assert abs(g1) < 0.01 * g1_single
    assert abs(g2) < 0.01 * g1_single


def test_clusterfield_shear_matches_bmo_delta_sigma(cfg, halo_model):
    """With miscentering and mass scatter switched (effectively) off, the
    cluster-field shear reduces to the analytic BMO DeltaSigma/Sigma_cr."""
    from snkappa.kappa import sigma_crit_msun_mpc2

    cl = pd.DataFrame([{"name": "c", "ra": 150.0, "dec": 30.0,
                        "z": 0.4, "m200": 2e14}])
    clf = ClusterField(cl, halo_model, mass_scatter_dex=1e-4,
                       miscenter_frac_r500=1e-4, conc=5.0)
    clf.set_zsrc(cfg.cosmo, 2.0)
    theta = 120.0
    k, g1, g2 = clf.kappa_shear_sum(150.0, 30.0 + theta / 3600.0)

    ib = halo_model.zbin_index([0.4])[0]
    rhoc = halo_model.rhoc[ib]
    c = 5.0
    r200 = (3 * 2e14 / (4 * np.pi * 200 * rhoc)) ** (1 / 3)
    rs = r200 / c
    rhos = 200 / 3 * rhoc * c ** 3 / (np.log(1 + c) - c / (1 + c))
    scr = sigma_crit_msun_mpc2(cfg.cosmo, [halo_model.zbins[ib]], 2.0)[0]
    x = (theta / 206264.806) * halo_model.da[ib] / rs
    gt_exp = rhos * rs * halo_model.delta_sigma_dimless(
        np.array([x]), np.array([c]))[0] / scr
    gt = np.hypot(g1, g2)
    assert gt == pytest.approx(gt_exp, rel=0.05)
    # cluster due N of the evaluation point... evaluation point due N of
    # cluster: separation vector along dec either way -> pure gamma1
    assert abs(g2) < 1e-10
    assert gt > k * 0.2  # sane relative scale at 120 arcsec