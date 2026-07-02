"""Split-normal p(z) reconstruction from photo-z quantiles."""

import numpy as np

from snkappa import photoz


def test_pdf_normalization_and_quantiles():
    zgrid = np.linspace(0.001, 6.0, 4000)
    mu, slo, shi = photoz.split_normal_params([0.8], [0.7], [0.95], [0.1])
    pdf = photoz.grid_pdf(zgrid, mu, slo, shi)[0]
    dz = zgrid[1] - zgrid[0]
    assert abs(pdf.sum() * dz - 1.0) < 1e-3
    cdf = np.cumsum(pdf) * dz
    z_l68 = zgrid[np.searchsorted(cdf, 0.16)]
    z_med = zgrid[np.searchsorted(cdf, 0.50)]
    z_u68 = zgrid[np.searchsorted(cdf, 0.84)]
    assert abs(z_med - 0.8) < 0.01
    assert abs(z_l68 - 0.7) < 0.02
    assert abs(z_u68 - 0.95) < 0.02


def test_truncated_grid_downweights_near_source():
    """A galaxy whose PDF straddles z_src keeps only its foreground mass."""
    z_src = 2.0
    zgrid = np.linspace(0.01, z_src - 0.01, 300)
    mu, slo, shi = photoz.split_normal_params([1.95], [1.75], [2.15], [0.2])
    w = photoz.grid_weights(zgrid, mu, slo, shi)[0]
    assert 0.3 < w.sum() < 0.7  # roughly half the PDF is background

    mu, slo, shi = photoz.split_normal_params([0.5], [0.45], [0.55], [0.05])
    w = photoz.grid_weights(zgrid, mu, slo, shi)[0]
    assert w.sum() > 0.98  # fully foreground galaxy keeps all its weight


def test_sampling_matches_pdf():
    rng = np.random.default_rng(11)
    mu, slo, shi = photoz.split_normal_params([0.6], [0.5], [0.8], [0.15])
    draws = photoz.sample(rng, mu, slo, shi, 20000)[:, 0]
    assert (draws > 0).all()
    assert abs(np.median(draws) - 0.6) < 0.01
    assert abs(np.quantile(draws, 0.16) - 0.5) < 0.02
    assert abs(np.quantile(draws, 0.84) - 0.8) < 0.02
