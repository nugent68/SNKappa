#!/usr/bin/env python
"""Latent-variable regression for the DES lensing amplitude (snkappa.latent).

Stage 1 (--noise): per-SN prediction-noise variances for EVERY sight line
via the single-SN Monte Carlo (photo-z + M* + SMHM + c joint draws) plus
the cluster-tier MC (mass scatter + miscentering), written to
output/des_full/latent_noise.csv. Unlike scripts/attenuation_mc.py (which
samples 120 sight lines to estimate a population lambda), this needs all
of them, so the HaloModel is built ONCE per field group and shared.

Stage 2 (default): empirical-Bayes errors-in-variables fit of the
amplitude -- single A and the (A_gal, A_cl) two-component version --
written to output/des_full/latent_fit.json. sigma_y is MUERR with the
catalog-visible lensing variance removed (A_fid^2 B^2 tau_b^2, avoiding
double counting against the latent term; --afid, default 0.79).

Run:  .venv/bin/python scripts/latent_fit.py --noise   (once, ~1-2 h)
      .venv/bin/python scripts/latent_fit.py           (fit, seconds)
"""

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

# per-SN engines share a z<=1.15 HaloModel; astropy warns (harmlessly,
# values are clipped) about lens bins beyond each SN's z_src
warnings.filterwarnings("ignore", message=".*Second redshift.*")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa import catalog, clusters as clu, montecarlo
from snkappa.datalab import TapClient
from snkappa.halos import HaloModel
from snkappa.kappa import KappaEngine, angular_sep_arcsec
from snkappa.latent import SLOPE_TH, fit_amplitude, fit_two_component
from snkappa.stellar import make_estimator

import des_full

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

NOISE_CSV = Path("output/des_full/latent_noise.csv")
FIT_JSON = Path("output/des_full/latent_fit.json")


def rvar(a):
    """Robust variance (half the 16-84 range squared): the cluster MC has
    a heavy miscentering tail that would otherwise dominate."""
    p16, p84 = np.percentile(a, [16, 84])
    return float((0.5 * (p84 - p16)) ** 2)


def stage_noise(args):
    res = pd.read_csv("output/des_full/des_all_kappa.csv")
    rng = np.random.default_rng(27182)
    rows = []
    for gname, (fields, center) in des_full.FIELD_GROUPS.items():
        gsn = res[res.GROUP == gname]
        if len(gsn) == 0:
            continue
        # region radius from the FULL group so the ADQL matches
        # scripts/des_full.py exactly and hits the TAP cache
        rad = angular_sep_arcsec(center[0], center[1],
                                 gsn.HOST_RA.to_numpy(),
                                 gsn.HOST_DEC.to_numpy()).max() / 3600.0
        if args.limit:
            gsn = gsn.iloc[:args.limit]

        class A:
            smhm_inverse = "posterior"; logmh_max = 13.8; n_rand = 500
        cfg0 = des_full.make_cfg(A, center, rad + 0.25)
        tap = TapClient(cfg0.data.tap_url, cfg0.data.cache_dir)
        df = catalog.clean_and_merge(cfg0, *catalog.fetch_regional(cfg0, tap))
        cl = clu.fetch_clusters(cfg0,
                                lambda u: TapClient(u, cfg0.data.cache_dir))
        # ONE HaloModel for the group (z_src=1.15 superset; per-SN engines
        # mask background bins via their own Sigma_crit)
        hm = HaloModel(cfg0.halo_model, cfg0.cosmo, 1.15)
        members = clu.assign_members(cfg0, df, cl, hm)
        est = make_estimator("nir1um_fsf", cfg0.cosmo)
        log(f"{gname}: catalog {len(df)}, {len(cl)} clusters, "
            f"{len(gsn)} SNe")

        r_ap = des_full.APERTURE_ARCMIN * 60.0
        for j, (_, sn) in enumerate(gsn.iterrows()):
            z_src = float(sn.zHD)
            theta = angular_sep_arcsec(sn.HOST_RA, sn.HOST_DEC,
                                       df.ra.to_numpy(), df.dec.to_numpy())
            in_ap = theta < r_ap + 30.0
            df_ap = df[in_ap].reset_index(drop=True)
            memb_ap = members[in_ap]

            cfg = des_full.make_cfg(A, (sn.HOST_RA, sn.HOST_DEC), rad + 0.25)
            cfg.source.z_src = z_src
            cfg.montecarlo.n_mc = args.n_mc
            eng = KappaEngine(cfg, cfg.cosmo, hm, est, df_ap)

            # CLASSICAL pass: only the components where the prediction is
            # built from a noisy MEASUREMENT (M* measurement error).
            # Photo-z widths collapsed (the prediction marginalizes p(z):
            # Berkson) and SMHM/c intrinsic scatter zeroed (posterior-mean
            # inversion / mean c-model: Berkson).
            cfg_cl = copy.deepcopy(cfg)
            cfg_cl.halo_model.smhm_scatter_dex = 1e-4
            cfg_cl.halo_model.c_scatter_dex = 1e-4
            slo, shi = eng.phot_slo, eng.phot_shi
            eng.phot_slo = np.full_like(slo, 1e-4)
            eng.phot_shi = np.full_like(shi, 1e-4)
            draws_cl = montecarlo.mc_kappa_raw(cfg_cl, eng, rng, memb_ap)
            eng.phot_slo, eng.phot_shi = slo, shi
            # TOTAL pass (classical + Berkson): all components
            draws_tot = montecarlo.mc_kappa_raw(cfg, eng, rng, memb_ap)

            cl_fg = cl[cl.z.to_numpy() < z_src - 0.02]
            ck = clu.ClusterKappa(cfg, hm, cfg.cosmo, cl_fg)
            nmc = min(args.n_mc, 48)
            # cluster classical = richness-mass measurement error only;
            # miscentering is marginalized in the prediction (Berkson)
            n_fg = len(ck.df)
            if n_fg:
                cl_cl = np.array([ck.kappa_sum(
                    sn.HOST_RA, sn.HOST_DEC,
                    dlogm=rng.normal(0.0, cfg.clusters.mass_scatter_dex,
                                     n_fg)) for _ in range(nmc)])
            else:
                cl_cl = np.zeros(nmc)
            cl_tot = ck.mc_kappa_sum(sn.HOST_RA, sn.HOST_DEC, rng, nmc)
            rows.append({
                "CID": sn.CID, "GROUP": gname, "zHD": z_src,
                "rvar_gal_cl": rvar(draws_cl),
                "rvar_gal_tot": rvar(draws_tot),
                "rvar_cl_cl": rvar(cl_cl), "rvar_cl_tot": rvar(cl_tot),
                "var_zp": (sn.rand_sig ** 2) / max(int(sn.n_rand_ok), 1)})
            if (j + 1) % 50 == 0:
                log(f"  {gname}: {j + 1}/{len(gsn)}")
        log(f"{gname}: done ({len(rows)} total)")
    pd.DataFrame(rows).to_csv(NOISE_CSV, index=False)
    log(f"saved {NOISE_CSV}")


def stage_fit(args):
    res = pd.read_csv("output/des_full/des_all_kappa.csv")
    noise = pd.read_csv(NOISE_CSV)
    d = res.merge(noise[["CID", "rvar_gal_cl", "rvar_gal_tot",
                         "rvar_cl_cl", "rvar_cl_tot", "var_zp"]], on="CID")
    d = d[d.PROBIA > 0.9].reset_index(drop=True)
    rng = np.random.default_rng(31415)
    log(f"fitting {len(d)} SNe (P(Ia)>0.9) with per-SN noise variances")

    zbin = d.zbin.to_numpy()
    # classical (shrinkage-driving) vs Berkson (residual-only) split
    s2_cl = (d.rvar_gal_cl + d.rvar_cl_cl + d.var_zp).to_numpy()
    s2_bk = (np.clip(d.rvar_gal_tot - d.rvar_gal_cl, 0, None)
             + np.clip(d.rvar_cl_tot - d.rvar_cl_cl, 0, None)).to_numpy()
    s2_all = s2_cl + s2_bk
    x = d.kappa_ext.to_numpy()
    y = d.hr.to_numpy()

    # sigma_y: MUERR minus the catalog-visible lensing variance already
    # modeled through the latent term (A_fid fixed; effect is small)
    tau2_apx = np.clip(d.rand_sig.to_numpy() ** 2 - s2_cl, 1e-10, None)
    sig2_y = np.clip(d.MUERR.to_numpy() ** 2
                     - (SLOPE_TH * args.afid) ** 2 * tau2_apx,
                     (0.5 * d.MUERR.to_numpy()) ** 2, None)

    # randoms-based tau2 fallback for thin bins
    rb = {zb: float(np.clip(
            d.loc[d.zbin == zb, "rand_sig"].mean() ** 2
            - s2_cl[zbin == zb].mean(), 1e-10, None))
          for zb in np.unique(zbin)}

    single = fit_amplitude(x, s2_cl, zbin, y, sig2_y, rng=rng,
                           n_boot=args.n_boot, rand_based_tau2=rb,
                           s2_berk=s2_bk)
    log(f"single: A = {single['A']:.3f} +- {single['A_err']:.3f} "
        f"(post {single['A_err_post']:.3f}, boot {single['A_err_boot']:.3f}; "
        f"<w> = {single['mean_reliability']:.2f})")

    # bracketing sensitivity: ALL MC variance treated as classical
    # (over-shrinks; upper bound on A)
    allcl = fit_amplitude(x, s2_all, zbin, y, sig2_y, rng=None,
                          rand_based_tau2=None)
    log(f"  bracket (all-classical): A = {allcl['A']:.3f} "
        f"+- {allcl['A_err_post']:.3f} (<w> = "
        f"{allcl['mean_reliability']:.2f})")

    # components: attach the (tiny) zero-point noise half-and-half
    two = fit_two_component(
        d.kappa_gal_ext.to_numpy(),
        (d.rvar_gal_cl + 0.5 * d.var_zp).to_numpy(),
        d.kappa_cl_ext.to_numpy(),
        (d.rvar_cl_cl + 0.5 * d.var_zp).to_numpy(),
        zbin, y, sig2_y, rng=rng, n_boot=max(args.n_boot // 2, 20),
        s2g_berk=np.clip(d.rvar_gal_tot - d.rvar_gal_cl, 0, None).to_numpy(),
        s2c_berk=np.clip(d.rvar_cl_tot - d.rvar_cl_cl, 0, None).to_numpy())
    log(f"two-comp: A_gal = {two['A_gal']:.3f} +- {two['A_gal_err']:.3f} | "
        f"A_cl = {two['A_cl']:.3f} +- {two['A_cl_err']:.3f} "
        f"(posterior corr {two['corr']:+.2f})")

    out = {"n_sn": len(d), "afid": args.afid, "n_boot": args.n_boot,
           "single": single, "single_all_classical_bracket": allcl,
           "two_component": two,
           "mean_s2_classical": float(s2_cl.mean()),
           "mean_s2_berkson": float(s2_bk.mean()),
           "note": ("empirical-Bayes EIV fit (snkappa.latent): Gaussian "
                    "per-z-bin latent prior; per-SN MC noise split into "
                    "classical (M* measurement, richness mass, zero "
                    "point; drives shrinkage) and Berkson (photo-z p(z), "
                    "SMHM/c intrinsic scatter, miscentering; residual "
                    "only); alpha profiled, flat prior on A; errors = "
                    "posterior std (+) bootstrap re-estimating tau_b")}
    FIT_JSON.write_text(json.dumps(out, indent=2, default=float))
    log(f"saved {FIT_JSON}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", action="store_true",
                    help="stage 1: compute per-SN noise variances (slow)")
    ap.add_argument("--n-mc", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0,
                    help="stage 1: max SNe per group (0 = all; for timing)")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--afid", type=float, default=0.79)
    args = ap.parse_args()
    if args.noise:
        stage_noise(args)
    else:
        stage_fit(args)


if __name__ == "__main__":
    main()
