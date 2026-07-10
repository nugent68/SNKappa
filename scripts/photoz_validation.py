#!/usr/bin/env python
"""Empirical validation of the reconstructed split-normal p(z) against DESI
spec-z (TODO 1.4).

Uses the DESI DR1 galaxies in the XMM-LSS (X) and Stripe 82 (S) regions as
truth: PIT (probability integral transform) / 68% coverage of the split-normal
p(z) pinned to the served LS-DR10 photo-z quantiles, and the
catastrophic-outlier rate, as functions of magnitude and redshift.

Caveat (reported, not hidden): DESI targets are brighter than the z<=22.5
catalog limit, so the faintest bins are sparsely sampled; numbers are quoted
per magnitude bin so the bright-end extrapolation is explicit.

Run: .venv/bin/python scripts/photoz_validation.py
Writes output/des_full/photoz_validation.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa import catalog, photoz
from snkappa.datalab import TapClient
from snkappa.kappa import angular_sep_arcsec  # noqa: F401 (parity import)

import des_full  # reuse the exact region definitions -> cache hits


def split_normal_cdf(z, mu, slo, shi):
    """CDF of the z>0-truncated split normal used by snkappa.photoz."""
    p_neg = 0.5 * (1.0 + erf(-mu / (np.sqrt(2.0) * slo)))
    lo = 0.5 * (1.0 + erf((z - mu) / (np.sqrt(2.0) * slo)))
    hi = 0.5 * erf((z - mu) / (np.sqrt(2.0) * shi))
    f = np.where(z <= mu, lo - p_neg, 0.5 - p_neg + hi)
    return np.clip(f / (1.0 - p_neg), 0.0, 1.0)


def stats_for(sel, zs, mu, slo, shi, magz):
    zs, mu, slo, shi, magz = (a[sel] for a in (zs, mu, slo, shi, magz))
    pit = split_normal_cdf(zs, mu, slo, shi)
    cov68 = float(np.mean((pit > 0.16) & (pit < 0.84)))
    cov95 = float(np.mean((pit > 0.025) & (pit < 0.975)))
    dz = (mu - zs) / (1.0 + zs)
    out = {
        "n": int(sel.sum()),
        "coverage_68": round(cov68, 4),
        "coverage_95": round(cov95, 4),
        "pit_mean": round(float(pit.mean()), 4),
        "sigma_nmad": round(float(1.4826 * np.median(
            np.abs(dz - np.median(dz)))), 4),
        "bias_med": round(float(np.median(dz)), 4),
        "outlier_rate_0.15": round(float(np.mean(np.abs(dz) > 0.15)), 4),
    }
    return out


def main():
    report = {}
    for gname in ("X", "S"):
        fields, center = des_full.FIELD_GROUPS[gname]
        sn = des_full.load_des()
        sn = sn[sn.FIELD.isin(fields)]
        rad = angular_sep_arcsec(center[0], center[1],
                                 sn.HOST_RA.to_numpy(),
                                 sn.HOST_DEC.to_numpy()).max() / 3600.0

        class A:                     # minimal args stand-in for make_cfg
            smhm_inverse = "posterior"; logmh_max = 13.8; n_rand = 500
        cfg = des_full.make_cfg(A, center, rad + 0.25)
        tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
        df = catalog.clean_and_merge(cfg, *catalog.fetch_regional(cfg, tap))

        m = df.z_spec.notna() & df.zp_med.notna() & (df.zp_med > 0)
        zs = df.z_spec.to_numpy(float)
        mu, slo, shi = photoz.split_normal_params(
            df.zp_med.to_numpy(float), df.zp_l68.to_numpy(float),
            df.zp_u68.to_numpy(float), df.zp_std.to_numpy(float))
        magz = df.mag_z.to_numpy(float)
        sel_all = m.to_numpy()

        g = {"all": stats_for(sel_all, zs, mu, slo, shi, magz)}
        for lo, hi in ((16, 20), (20, 21), (21, 21.75), (21.75, 22.5)):
            g[f"magz_{lo}-{hi}"] = stats_for(
                sel_all & (magz >= lo) & (magz < hi), zs, mu, slo, shi, magz)
        for lo, hi in ((0.0, 0.4), (0.4, 0.8), (0.8, 1.15)):
            g[f"zspec_{lo}-{hi}"] = stats_for(
                sel_all & (zs >= lo) & (zs < hi), zs, mu, slo, shi, magz)
        report[gname] = g
        print(f"== {gname} ==")
        for k, v in g.items():
            print(f"  {k:18s} N={v['n']:6d} cov68={v['coverage_68']:.3f} "
                  f"cov95={v['coverage_95']:.3f} nmad={v['sigma_nmad']:.4f} "
                  f"bias={v['bias_med']:+.4f} "
                  f"outlier={v['outlier_rate_0.15']:.3f}")

    out = Path("output/des_full/photoz_validation.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
