"""Random sightlines: mean-field zero point, empirical LOS variance, and the
weighted-number-count overdensity zeta (H0LiCOW/TDCOSMO-style diagnostic).

Randoms are drawn from the SAME regional catalog as the science sightline
(annulus around the SN), so depth and coverage are matched by construction.
Each random applies the same central exclusion disk (r_exclude) as the SN
sightline, for an apples-to-apples estimator. Note: apertures of nearby
randoms overlap, so the effective number of independent patches is smaller
than n_random_los; the empirical scatter is a mild underestimate of extreme
tails (documented caveat).
"""

from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u


def draw_random_centers(cfg, rng, n=None):
    """Uniform points in the control annulus around the SN (proper spherical
    offsets via astropy)."""
    n = n or cfg.randoms.n_random_los
    r1, r2 = cfg.randoms.annulus_deg
    r = np.sqrt(rng.uniform(r1**2, r2**2, n))
    pa = rng.uniform(0.0, 360.0, n)
    center = SkyCoord(cfg.source.ra_src * u.deg, cfg.source.dec_src * u.deg)
    pts = center.directional_offset_by(pa * u.deg, r * u.deg)
    return pts.ra.deg, pts.dec.deg


def run_randoms(cfg, engine, rng, progress=None):
    """kappa_raw and weighted counts for each random sightline.

    Returns dict with per-random arrays and summary stats. Randoms whose
    aperture counts fall below half the median (masking / survey edge) are
    flagged and excluded from the zero point.
    """
    ras, decs = draw_random_centers(cfg, rng)
    r_ap = cfg.los.aperture_radius_arcmin * 60.0
    r_in = cfg.deflector.r_exclude_arcsec

    kappa = np.empty(len(ras))
    ngal = np.empty(len(ras), dtype=int)
    zeta_counts = {r: np.empty((len(ras), 3)) for r in
                   cfg.los.count_aperture_arcsec}

    for i, (ra0, dec0) in enumerate(zip(ras, decs)):
        _, _, k = engine.kappa_los(ra0, dec0, r_in, r_ap)
        kappa[i] = k.sum()
        ngal[i] = k.size
        for r_c in cfg.los.count_aperture_arcsec:
            zeta_counts[r_c][i] = engine.weighted_counts(ra0, dec0, r_in, r_c)
        if progress and (i + 1) % 100 == 0:
            progress(f"  randoms: {i + 1}/{len(ras)}")

    ok = ngal > 0.5 * np.median(ngal)
    k_ok = kappa[ok]
    p16, p50, p84 = np.percentile(k_ok, [16, 50, 84])
    out = {
        "ra": ras, "dec": decs, "kappa_raw": kappa, "ngal": ngal, "ok": ok,
        "n_flagged": int((~ok).sum()),
        "kappa_mean": float(k_ok.mean()),
        "kappa_median": float(p50),
        "kappa_std": float(k_ok.std(ddof=1)),
        "kappa_sigma_robust": float(0.5 * (p84 - p16)),
        "counts": zeta_counts,
    }
    return out


def zeta_summary(cfg, engine, randoms, exclude_mask):
    """zeta = counts_LOS / median(counts_random) for each count aperture and
    weighting scheme (unweighted, 1/r, lensing-efficiency)."""
    r_in = cfg.deflector.r_exclude_arcsec
    schemes = ["unweighted", "inv_r", "efficiency"]
    out = {}
    for r_c in cfg.los.count_aperture_arcsec:
        sn = engine.weighted_counts(cfg.source.ra_src, cfg.source.dec_src,
                                    r_in, r_c, exclude_mask=exclude_mask)
        rand = randoms["counts"][r_c][randoms["ok"]]
        entry = {}
        for j, scheme in enumerate(schemes):
            med = float(np.median(rand[:, j]))
            entry[scheme] = {
                "sn": float(sn[j]),
                "random_median": med,
                "zeta": float(sn[j] / med) if med > 0 else np.nan,
                "percentile_among_randoms": float(
                    100.0 * np.mean(rand[:, j] < sn[j])),
            }
        out[f"r_{r_c:g}_arcsec"] = entry
    return out
