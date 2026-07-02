import numpy as np
import pandas as pd
import pytest

from snkappa.config import (Config, CosmologyConfig, DataConfig,
                            DeflectorConfig, HaloModelConfig, LensGroupConfig,
                            LosConfig, MonteCarloConfig, OutputConfig,
                            RandomsConfig, SourceConfig)


@pytest.fixture(scope="session")
def cfg():
    return Config(
        source=SourceConfig(name="TEST", ra_src=150.0, dec_src=30.0, z_src=2.0),
        deflector=DeflectorConfig(ra_lens=150.0, dec_lens=30.0, z_lens=0.4,
                                  theta_E_arcsec=1.5),
        lens_group=LensGroupConfig(),
        los=LosConfig(aperture_radius_arcmin=10.0),
        randoms=RandomsConfig(n_random_los=50, annulus_deg=[0.3, 0.8]),
        montecarlo=MonteCarloConfig(n_mc=100, seed=1234),
        data=DataConfig(),
        halo_model=HaloModelConfig(),
        cosmology=CosmologyConfig(),
        output=OutputConfig(),
    )


@pytest.fixture(scope="session")
def halo_model(cfg):
    from snkappa.halos import HaloModel
    return HaloModel(cfg.halo_model, cfg.cosmo, cfg.source.z_src, dz=0.05)


def synthetic_catalog(rng, n, ra0, dec0, radius_deg, z_src, frac_spec=0.3):
    """Uniform random galaxies with plausible photometry."""
    r = radius_deg * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    dec = dec0 + r * np.sin(phi)
    ra = ra0 + r * np.cos(phi) / np.cos(np.radians(dec0))
    z_true = rng.uniform(0.05, z_src - 0.1, n)
    is_spec = rng.random(n) < frac_spec
    mag_z = rng.uniform(17.0, 21.0, n)
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z_spec": np.where(is_spec, z_true, np.nan),
        "zp_med": z_true + rng.normal(0, 0.05, n),
        "zp_std": np.full(n, 0.08),
        "mag_g": mag_z + 1.4, "mag_r": mag_z + 0.7,
        "mag_i": mag_z + 0.3, "mag_z": mag_z,
        "excl_deflector": np.zeros(n, dtype=bool),
        "is_group": np.zeros(n, dtype=bool),
    })
    df["zp_l68"] = df["zp_med"] - 0.08
    df["zp_u68"] = df["zp_med"] + 0.08
    return df
