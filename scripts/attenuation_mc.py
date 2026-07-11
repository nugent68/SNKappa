#!/usr/bin/env python
"""Errors-in-variables attenuation of the lensing slope (TODO 2.1).

Noise in the PREDICTED kappa (photo-z, M*/L scatter, SMHM scatter,
concentration scatter, cluster mass scatter) attenuates a regression of
Hubble residuals on kappa_pred:

    A_fit = A_true * lambda,   lambda = 1 - <Var_noise(kappa)> / Var(kappa_ext)

This script measures per-SN Var_noise(kappa) with the package's single-SN
Monte Carlo (snkappa.montecarlo.mc_kappa_raw: joint photo-z + M* + SMHM + c
draws) on a stratified subsample of DES sightlines, plus the cluster-tier MC
(mass scatter + miscentering) where clusters are nearby, plus the per-bin
zero-point sampling noise sigma_rand^2 / N_rand. Reports lambda and the
de-attenuated amplitude A_fit / lambda.

Run AFTER the headline scripts/des_full.py:
    .venv/bin/python scripts/attenuation_mc.py [--n-per-group 30] [--n-mc 64]
Writes output/des_full/attenuation.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa import catalog, clusters as clu, montecarlo
from snkappa.datalab import TapClient
from snkappa.halos import HaloModel
from snkappa.kappa import KappaEngine, angular_sep_arcsec
from snkappa.stellar import make_estimator

import des_full

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-group", type=int, default=30)
    ap.add_argument("--n-mc", type=int, default=64)
    args = ap.parse_args()

    res = pd.read_csv("output/des_full/des_all_kappa.csv")
    good = res[res.PROBIA > 0.9].reset_index(drop=True)
    rng = np.random.default_rng(31415)

    rows = []
    for gname, (fields, center) in des_full.FIELD_GROUPS.items():
        gsn = good[good.GROUP == gname]
        if len(gsn) == 0:
            continue
        # stratified subsample: uniform in redshift rank, so the noise
        # variance is sampled across the whole lensing-kernel range
        take = gsn.sort_values("zHD").iloc[
            np.unique(np.linspace(0, len(gsn) - 1,
                                  args.n_per_group).astype(int))]

        # region radius from ALL SNe in the group (not the P>0.9 subset), so
        # the ADQL matches scripts/des_full.py exactly and hits the TAP cache
        allsn = res[res.GROUP == gname]
        rad = angular_sep_arcsec(center[0], center[1],
                                 allsn.HOST_RA.to_numpy(),
                                 allsn.HOST_DEC.to_numpy()).max() / 3600.0

        class A:
            smhm_inverse = "posterior"; logmh_max = 13.8; n_rand = 500
        cfg0 = des_full.make_cfg(A, center, rad + 0.25)
        tap = TapClient(cfg0.data.tap_url, cfg0.data.cache_dir)
        df = catalog.clean_and_merge(cfg0, *catalog.fetch_regional(cfg0, tap))
        cl = clu.fetch_clusters(cfg0,
                                lambda u: TapClient(u, cfg0.data.cache_dir))
        hm_full = HaloModel(cfg0.halo_model, cfg0.cosmo, 1.15)
        members = clu.assign_members(cfg0, df, cl, hm_full)
        est = make_estimator("nir1um_fsf", cfg0.cosmo)
        log(f"{gname}: catalog {len(df)}, {len(cl)} clusters; "
            f"{len(take)} subsample SNe")

        r_ap = des_full.APERTURE_ARCMIN * 60.0
        for _, sn in take.iterrows():
            z_src = float(sn.zHD)
            theta = angular_sep_arcsec(sn.HOST_RA, sn.HOST_DEC,
                                       df.ra.to_numpy(), df.dec.to_numpy())
            in_ap = theta < r_ap + 30.0
            df_ap = df[in_ap].reset_index(drop=True)
            memb_ap = members[in_ap]

            cfg = des_full.make_cfg(A, (sn.HOST_RA, sn.HOST_DEC), rad + 0.25)
            cfg.source.z_src = z_src
            cfg.montecarlo.n_mc = args.n_mc
            hm = HaloModel(cfg.halo_model, cfg.cosmo, z_src)
            eng = KappaEngine(cfg, cfg.cosmo, hm, est, df_ap)
            draws = montecarlo.mc_kappa_raw(cfg, eng, rng, memb_ap)

            # cluster-tier variance (mass scatter + miscentering)
            cl_fg = cl[cl.z.to_numpy() < z_src - 0.02]
            ck = clu.ClusterKappa(cfg, hm, cfg.cosmo, cl_fg)
            cl_draws = ck.mc_kappa_sum(sn.HOST_RA, sn.HOST_DEC, rng,
                                       min(args.n_mc, 48))

            var_zp = (sn.rand_sig ** 2) / max(int(sn.n_rand_ok), 1)

            def rvar(a):
                """Robust variance: (half the 16-84 range)^2. The raw MC
                variance of the cluster term is dominated by rare
                miscentering draws that land the cluster core on the
                sightline -- a heavy tail that breaks the Gaussian
                errors-in-variables interpretation."""
                p16, p84 = np.percentile(a, [16, 84])
                return (0.5 * (p84 - p16)) ** 2

            v = rvar(draws) + rvar(cl_draws) + var_zp
            rows.append({"CID": sn.CID, "GROUP": gname, "zHD": z_src,
                         "kappa_ext": sn.kappa_ext,
                         "var_gal": float(np.var(draws)),
                         "var_cl": float(np.var(cl_draws)),
                         "rvar_gal": rvar(draws), "rvar_cl": rvar(cl_draws),
                         "var_zp": var_zp, "var_noise": v})
        log(f"{gname}: done ({len(rows)} total)")

    sub = pd.DataFrame(rows)
    var_x = float(good.kappa_ext.var())
    mean_noise = float(sub.var_noise.mean())
    med_noise = float(sub.var_noise.median())
    lam_mean = max(1e-3, 1.0 - mean_noise / var_x)
    lam_med = max(1e-3, 1.0 - med_noise / var_x)
    out = {
        "n_sub": len(sub), "n_mc": args.n_mc,
        "var_kappa_ext_population": var_x,
        "mean_var_noise": mean_noise, "median_var_noise": med_noise,
        "lambda_mean": lam_mean, "lambda_median": lam_med,
        "note": ("A_true = A_fit / lambda; components are galaxy-halo MC "
                 "(photo-z, M*, SMHM, c), cluster MC (mass, miscentering), "
                 "and zero-point sampling noise"),
        "per_sn": rows,
    }
    Path("output/des_full/attenuation.json").write_text(
        json.dumps(out, indent=2, default=float))
    log(f"Var(kappa_ext) = {var_x:.3e}")
    log(f"<Var_noise> mean/median = {mean_noise:.3e} / {med_noise:.3e}")
    log(f"lambda (mean/median) = {lam_mean:.3f} / {lam_med:.3f}")
    log("saved output/des_full/attenuation.json")


if __name__ == "__main__":
    main()
