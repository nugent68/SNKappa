"""Survey-region construction for multi-field SN samples.

Factored from the Union3 feasibility audit: friends-of-friends grouping
of sight lines into survey regions with wrap-safe spherical-mean centers,
plus the mixed-convention coordinate parser (some compilations serve RA
as sexagesimal HOURS, others as decimal DEGREES in the same list --
parsing decimals as hourangle scatters fields by x15 mod 360).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

from .kappa import angular_sep_arcsec


def parse_mixed_coords(ra_list, dec_list):
    """(ra_deg, dec_deg) arrays from per-entry mixed conventions:
    colon-separated => sexagesimal (RA in hours), else decimal degrees."""
    ra = np.array([
        float(SkyCoord(r, d, unit=(u.hourangle, u.deg)).ra.deg)
        if ":" in str(r) else float(r)
        for r, d in zip(ra_list, dec_list)])
    dec = np.array([
        float(SkyCoord("0h", d, unit=(u.hourangle, u.deg)).dec.deg)
        if ":" in str(d) else float(d) for d in dec_list])
    return ra, dec


def fof(ra, dec, link_deg):
    """Friends-of-friends group labels on the sphere (O(N^2); fine for
    catalog-scale N)."""
    n = len(ra)
    labels = -np.ones(n, dtype=int)
    g = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = g
        while stack:
            j = stack.pop()
            sep = angular_sep_arcsec(ra[j], dec[j], ra, dec) / 3600.0
            for k in np.flatnonzero((sep < link_deg) & (labels < 0)):
                labels[k] = g
                stack.append(k)
        g += 1
    return labels


def spherical_center(ra, dec):
    """Wrap-safe mean position (unit-vector mean)."""
    c = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    v = c.cartesian.xyz.value.mean(axis=1)
    s = SkyCoord(x=v[0], y=v[1], z=v[2],
                 representation_type="cartesian").spherical
    return float(s.lon.deg) % 360.0, float(s.lat.deg)


def build_regions(ra, dec, z, link_deg=0.5, min_n=2, z_min_singleton=0.2):
    """Group sight lines into survey regions.

    Returns (labels, regions) where labels aligns with the inputs and
    regions is a DataFrame [region, ra, dec, radius_deg, n_sn, z_max,
    process] -- `process` False for singletons below z_min_singleton
    (negligible lensing weight; the caller logs the loss).
    """
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    z = np.asarray(z, float)
    labels = fof(ra, dec, link_deg)
    rows = []
    for gname in np.unique(labels):
        m = labels == gname
        ra0, dec0 = spherical_center(ra[m], dec[m])
        rad = float(np.max(angular_sep_arcsec(
            ra0, dec0, ra[m], dec[m])) / 3600.0)
        n = int(m.sum())
        rows.append({
            "region": int(gname), "ra": ra0, "dec": dec0,
            "radius_deg": rad, "n_sn": n,
            "z_max": float(z[m].max()),
            "process": bool(n >= min_n or z[m].max() >= z_min_singleton)})
    return labels, pd.DataFrame(rows)
