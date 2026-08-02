"""Region construction: FoF grouping, wrap-safe centers, and the
mixed-convention coordinate parser (the D3-at-hourangle bug class)."""

import numpy as np

from snkappa.regions import build_regions, parse_mixed_coords


def test_mixed_convention_parsing():
    """Sexagesimal entries are hours; plain decimals are degrees --
    SNLS D3 (214.5 deg) must NOT land at 337 deg."""
    ra, dec = parse_mixed_coords(
        ["14:17:54.0", "214.475", "02:26:00"],
        ["+52:30:00", "52.5", "-04:30:00"])
    assert abs(ra[0] - 214.475) < 0.01
    assert abs(ra[1] - 214.475) < 0.01      # decimal degrees untouched
    assert abs(ra[2] - 36.5) < 0.01
    assert abs(dec[0] - 52.5) < 0.01
    assert abs(dec[2] + 4.5) < 0.01


def test_build_regions_pointings_stripe_and_singletons():
    rng = np.random.default_rng(3)
    # two 0.3-deg pointings, a 6-deg stripe chunked by 0.5-deg linking,
    # and two isolated singletons (one low-z, one high-z)
    ra = np.concatenate([
        36.0 + rng.uniform(-0.15, 0.15, 20),
        214.5 + rng.uniform(-0.15, 0.15, 15),
        np.arange(310.0, 316.0, 0.15),       # contiguous stripe
        [100.0], [200.0]])
    dec = np.concatenate([
        -4.5 + rng.uniform(-0.15, 0.15, 20),
        52.5 + rng.uniform(-0.15, 0.15, 15),
        np.zeros(40), [10.0], [-10.0]])
    z = np.concatenate([np.full(20, 0.8), np.full(15, 0.6),
                        np.full(40, 0.3), [0.12], [0.5]])
    labels, reg = build_regions(ra, dec, z, link_deg=0.5,
                                min_n=2, z_min_singleton=0.2)
    assert len(labels) == len(ra)
    # the two pointings are single compact regions
    p1 = reg[(abs(reg.ra - 36.0) < 0.5)]
    assert len(p1) == 1 and p1.n_sn.iloc[0] == 20
    assert p1.radius_deg.iloc[0] < 0.4
    # the contiguous stripe percolates into ONE region at this linking
    stripe = reg[(reg.dec.abs() < 0.2) & (reg.n_sn > 1)]
    assert stripe.n_sn.sum() == 40
    # low-z singleton flagged non-process; high-z singleton kept
    lo = reg[(abs(reg.ra - 100.0) < 0.1)]
    hi = reg[(abs(reg.ra - 200.0) < 0.1)]
    assert not lo.process.iloc[0]
    assert hi.process.iloc[0]


def test_wrap_safe_center():
    ra = np.array([359.8, 0.2, 0.0])
    dec = np.array([0.0, 0.0, 0.1])
    _, reg = build_regions(ra, dec, np.full(3, 0.5), link_deg=1.0)
    assert len(reg) == 1
    assert reg.ra.iloc[0] < 1.0 or reg.ra.iloc[0] > 359.0
    assert reg.radius_deg.iloc[0] < 0.5
