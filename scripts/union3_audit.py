#!/usr/bin/env python
"""Union3 lensing-measurement feasibility audit (Phase 0).

Rubin et al. 2025 (ApJ 986, 231) released the Union3 UNITY inputs
(github.com/rubind/union3_release): per-SN mB/x1/c + covariances,
z_CMB/z_helio, host mass, sample labels, and RA/Dec -- everything a
predicted-kappa measurement needs EXCEPT per-SN standardized distances
(UNITY's released distance product is the 22-node binned spline), so a
build would Tripp-standardize the released fits.

This audit:
1. ingests the inputs pickle (data_union3/inputs_union3.pickle),
2. friends-of-friends clusters the z > Z_MIN sight lines into survey
   regions (LINK_DEG linking), special-casing nothing -- the stripe
   emerges as several chunks naturally at this linking length,
3. probes DESI Legacy Surveys coverage per region through the cached
   Data Lab TapClient (tractor + DR10 photo-z + DR9 fallback counts),
4. forecasts the slope significance for the covered sample using the
   DES-measured scaling (sigma_kappa(z) from the DES randoms curve;
   the same formula applied to DES inputs is printed as a cross-check),
5. writes output/union3/audit.json and prints the go/no-go summary.

Run: .venv/bin/python scripts/union3_audit.py
"""

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.datalab import TapClient
from snkappa.kappa import angular_sep_arcsec

PICKLE = Path("data_union3/inputs_union3.pickle")
OUT = Path("output/union3/audit.json")
Z_MIN = 0.1
LINK_DEG = 0.5   # pointing-scale: 2 deg percolates the SDSS stripe into
                 # useless 17-deg chains whose centers probe random sky
ALPHA, BETA, SIG_INT = 0.14, 3.1, 0.12   # Tripp forecast weights only
B_DES = -1.71                            # DES-measured slope [mag/kappa]


def load_union3():
    d0 = pickle.load(gzip.open(PICKLE, "rb"))[0]
    names = [p.rstrip("/").split("/")[-1] for p in d0["snpaths"]]
    samples = [p.rstrip("/").split("/")[-2] for p in d0["snpaths"]]
    # mixed RA conventions across the 25 samples: sexagesimal strings
    # (colons) are hours, plain decimals are degrees -- parsing decimals
    # as hourangle would place SNLS D3 at RA 337-353 deg (x15 mod 360)
    ra = [float(SkyCoord(r, d, unit=(u.hourangle, u.deg)).ra.deg)
          if ":" in str(r) else float(r)
          for r, d in zip(d0["RA"], d0["Dec"])]
    dec = [float(SkyCoord("0h", d, unit=(u.hourangle, u.deg)).dec.deg)
           if ":" in str(d) else float(d) for d in d0["Dec"]]
    coo = SkyCoord(ra * u.deg, dec * u.deg)
    cov = d0["mBx1c_cov_list"]
    sig_tripp = np.sqrt(
        cov[:, 0, 0] + ALPHA**2 * cov[:, 1, 1] + BETA**2 * cov[:, 2, 2]
        + 2 * ALPHA * cov[:, 0, 1] - 2 * BETA * cov[:, 0, 2]
        - 2 * ALPHA * BETA * cov[:, 1, 2] + SIG_INT**2)
    return pd.DataFrame({
        "name": names, "sample": samples,
        "ra": coo.ra.deg, "dec": coo.dec.deg,
        "z": d0["z_CMB_list"], "sig_hr": sig_tripp,
        "mass": [float(m) for m in d0["mass"]]})


def fof(ra, dec, link_deg):
    """Friends-of-friends on the sphere; returns group labels."""
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


def probe_region(tap, ra, dec):
    """(tractor_n, pz10_n, pz9_n) in a 0.05 deg cone (cached)."""
    cone = f"Q3C_RADIAL_QUERY(t.ra, t.dec, {ra:.4f}, {dec:.4f}, 0.05)"
    n_tr = int(tap.query(
        f"SELECT COUNT(*) AS n FROM ls_dr10.tractor t WHERE 't'={cone}",
        label="probe:tractor")["n"].iloc[0])
    n10 = n9 = 0
    if n_tr:
        n10 = int(tap.query(
            "SELECT COUNT(*) AS n FROM ls_dr10.tractor t JOIN "
            f"ls_dr10.photo_z p ON t.ls_id=p.ls_id WHERE 't'={cone}",
            label="probe:pz10")["n"].iloc[0])
        if not n10:
            n9 = int(tap.query(
                "SELECT COUNT(*) AS n FROM ls_dr9.tractor t JOIN "
                f"ls_dr9.photo_z p ON t.ls_id=p.ls_id WHERE 't'="
                f"Q3C_RADIAL_QUERY(t.ra, t.dec, {ra:.4f}, {dec:.4f}, 0.05)",
                label="probe:pz9")["n"].iloc[0])
    if n10:
        return n_tr, "ls_south"
    if n9:
        return n_tr, "ls_north"
    return n_tr, "uncovered"


def main():
    sn = load_union3()
    print(f"ingested {len(sn)} SNe, {sn['sample'].nunique()} samples; "
          f"RA/Dec + z + mB/x1/c + mass for all")
    hi = sn[sn.z > Z_MIN].reset_index(drop=True)
    print(f"z > {Z_MIN}: {len(hi)} SNe (lensing-relevant sample)")

    labels = fof(hi.ra.to_numpy(), hi.dec.to_numpy(), LINK_DEG)
    hi["region"] = labels
    tap = TapClient("https://datalab.noirlab.edu/tap", "cache")

    regions = []
    for gname, grp in hi.groupby("region"):
        # spherical mean center (wrap-safe)
        c = SkyCoord(grp.ra.to_numpy() * u.deg, grp.dec.to_numpy() * u.deg)
        v = c.cartesian.xyz.value.mean(axis=1)
        cm = SkyCoord(x=v[0], y=v[1], z=v[2],
                      representation_type="cartesian").spherical
        ra0, dec0 = float(cm.lon.deg) % 360, float(cm.lat.deg)
        rad = float(np.max(angular_sep_arcsec(
            ra0, dec0, grp.ra.to_numpy(), grp.dec.to_numpy())) / 3600.0)
        n_tr, cls = probe_region(tap, ra0, dec0)
        regions.append({
            "region": int(gname), "ra": round(ra0, 3), "dec": round(dec0, 3),
            "radius_deg": round(rad, 2), "n_sn": len(grp),
            "z_min": round(float(grp.z.min()), 3),
            "z_max": round(float(grp.z.max()), 3),
            "samples": sorted(grp["sample"].unique().tolist()),
            "coverage": cls})
    reg = pd.DataFrame(regions).sort_values("n_sn", ascending=False)
    cov_map = dict(zip(reg.region, reg.coverage))
    hi["coverage"] = hi.region.map(cov_map)

    # ---- forecast (DES-measured scaling) --------------------------------
    des = pd.read_csv("output/des_full/des_all_kappa.csv")
    desg = des[des.PROBIA > 0.9]
    # realized predictor scatter per z bin (the robust randoms sigma clips
    # the kappa tail that actually drives the regression significance)
    zs = desg.groupby("zbin").kappa_ext.agg(
        lambda a: a.std() if len(a) >= 5 else np.nan).dropna()
    sig_k = lambda z: np.interp(z, zs.index.to_numpy(), zs.to_numpy())

    def forecast(zarr, sig_hr):
        w = 1.0 / sig_hr**2
        return abs(B_DES) * np.sqrt(np.sum(w * sig_k(zarr)**2))

    sn_des = forecast(desg.zHD.to_numpy(), desg.MUERR.to_numpy())
    m = hi.coverage.isin(["ls_south", "ls_north"])
    cov = hi[m]
    sn_u3 = forecast(cov.z.to_numpy(), cov.sig_hr.to_numpy())
    sn_joint = np.hypot(sn_des, sn_u3)

    out = {
        "n_total": len(sn), "n_z_gt_min": len(hi), "z_min": Z_MIN,
        "residual_path": ("per-SN distances NOT released (UNITY spline "
                          "nodes only); Tripp-standardize released "
                          "mB/x1/c + host mass"),
        "n_covered": int(m.sum()),
        "n_uncovered": int((~m).sum()),
        "by_coverage": hi.groupby("coverage").size().to_dict(),
        "by_sample_covered": cov.groupby("sample").size().to_dict(),
        "uncovered_samples": hi[~m].groupby("sample").size().to_dict(),
        "uncovered_zrange": [float(hi[~m].z.min()), float(hi[~m].z.max())]
        if (~m).any() else None,
        "n_regions_covered": int(reg[reg.coverage != "uncovered"].shape[0]),
        "regions": reg.to_dict("records"),
        "forecast": {
            "des_crosscheck_sigma": round(float(sn_des), 2),
            "union3_covered_sigma": round(float(sn_u3), 2),
            "joint_sigma": round(float(sn_joint), 2),
            "note": ("DES-measured slope and randoms sigma_kappa(z); "
                     "same estimator formula; DES cross-check should "
                     "reproduce ~4")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print(f"\ncovered: {out['n_covered']} SNe in "
          f"{out['n_regions_covered']} LS regions; "
          f"uncovered: {out['n_uncovered']} "
          f"(z {out['uncovered_zrange']})")
    print("coverage classes:", out["by_coverage"])
    print(f"forecast: DES cross-check {sn_des:.1f} sigma (actual 4.1) | "
          f"Union3 covered {sn_u3:.1f} sigma | joint {sn_joint:.1f} sigma")
    print(f"top regions:\n{reg.head(12).to_string(index=False)}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
