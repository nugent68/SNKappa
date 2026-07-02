"""Configuration loading, validation, and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
from astropy.cosmology import FlatLambdaCDM


@dataclass
class SourceConfig:
    name: str
    ra_src: float
    dec_src: float
    z_src: float


@dataclass
class DeflectorConfig:
    ra_lens: float
    dec_lens: float
    z_lens: float
    theta_E_arcsec: float | None = None
    r_exclude_arcsec: float | None = None

    def __post_init__(self):
        if self.r_exclude_arcsec is None:
            if self.theta_E_arcsec is None:
                raise ValueError("Provide theta_E_arcsec or r_exclude_arcsec")
            self.r_exclude_arcsec = 5.0 * self.theta_E_arcsec


@dataclass
class LensGroupConfig:
    include_lens_group: bool = False
    r_group_arcmin: float = 2.0
    dz_group: float | None = None  # resolved against z_lens in Config.__post_init__


@dataclass
class LosConfig:
    aperture_radius_arcmin: float = 15.0
    count_aperture_arcsec: list = field(default_factory=lambda: [45.0, 120.0])
    mag_limit: float = 21.0        # dereddened AB mag limit in mag_limit_band
    mag_limit_band: str = "z"      # z-band: present in both LS north and south


@dataclass
class RandomsConfig:
    n_random_los: int = 500
    annulus_deg: list = field(default_factory=lambda: [0.5, 2.0])


@dataclass
class MonteCarloConfig:
    n_mc: int = 1000
    seed: int = 42


@dataclass
class DataConfig:
    desi_release: str = "dr1"
    ls_release: str = "dr10"
    tap_url: str = "https://datalab.noirlab.edu/tap"
    cache_dir: str = "cache"


@dataclass
class HaloModelConfig:
    mstar_method: str = "taylor2011"
    mstar_scatter_dex: float = 0.15   # M*/L (color-based estimator) scatter
    smhm: str = "behroozi13"          # or "moster13"
    smhm_scatter_dex: float = 0.18
    cmodel: str = "diemer19"
    c_scatter_dex: float = 0.16
    profile: str = "bmo"              # "bmo" (truncated NFW) or "nfw"
    mdef: str = "200c"


@dataclass
class CosmologyConfig:
    H0: float = 70.0
    Om0: float = 0.3


@dataclass
class OutputConfig:
    dir: str = "output/run"


@dataclass
class Config:
    source: SourceConfig
    deflector: DeflectorConfig
    lens_group: LensGroupConfig = field(default_factory=LensGroupConfig)
    los: LosConfig = field(default_factory=LosConfig)
    randoms: RandomsConfig = field(default_factory=RandomsConfig)
    montecarlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    data: DataConfig = field(default_factory=DataConfig)
    halo_model: HaloModelConfig = field(default_factory=HaloModelConfig)
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        if self.lens_group.dz_group is None:
            self.lens_group.dz_group = 0.005 * (1.0 + self.deflector.z_lens)

    @property
    def cosmo(self) -> FlatLambdaCDM:
        return FlatLambdaCDM(H0=self.cosmology.H0, Om0=self.cosmology.Om0)

    def as_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


_SECTIONS = {
    "source": SourceConfig,
    "deflector": DeflectorConfig,
    "lens_group": LensGroupConfig,
    "los": LosConfig,
    "randoms": RandomsConfig,
    "montecarlo": MonteCarloConfig,
    "data": DataConfig,
    "halo_model": HaloModelConfig,
    "cosmology": CosmologyConfig,
    "output": OutputConfig,
}


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    kwargs = {}
    for key, cls in _SECTIONS.items():
        if key in raw:
            section = {k: v for k, v in raw[key].items() if v is not None}
            kwargs[key] = cls(**section)
    return Config(**kwargs)
