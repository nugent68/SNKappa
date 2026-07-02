"""snkappa CLI: `python -m snkappa {check,run} --config <yaml>`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="snkappa",
        description="External-convergence estimator for strongly lensed SNe")
    parser.add_argument("command", choices=["check", "run"],
                        help="check: STEP-0 availability report only; "
                             "run: full pipeline")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args(argv)

    from .config import load_config
    from .datalab import TapClient, availability_report

    cfg = load_config(args.config)
    tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)

    report = availability_report(tap, cfg)
    if args.command == "check":
        return 0 if report["ok"] else 1
    if not report["ok"]:
        print("STEP 0 failed; aborting run.", file=sys.stderr)
        return 1
    return run_pipeline(cfg, tap, report)


def run_pipeline(cfg, tap, report) -> int:
    from . import catalog, corrections, montecarlo, outputs, plots, randoms
    from .halos import HaloModel
    from .kappa import KappaEngine
    from .stellar import make_estimator

    t0 = time.time()
    rng = np.random.default_rng(cfg.montecarlo.seed)
    outdir = Path(cfg.output.dir)
    outdir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # --- catalogs ------------------------------------------------------------
    log("fetching regional catalogs (cached after first run) ...")
    df_raw, zpix = catalog.fetch_regional(cfg, tap)
    log(f"  tractor rows: {len(df_raw)}, zpix rows: {len(zpix)}")
    df = catalog.clean_and_merge(cfg, df_raw, zpix)
    n_spec = int(df["z_spec"].notna().sum())
    n_phot = len(df) - n_spec
    log(f"  cleaned catalog: {len(df)} galaxies "
        f"({n_spec} spec-z, {n_phot} photo-z, "
        f"{df.attrs['n_dropped_no_z']} dropped for no usable z)")

    frac_ok, n_flag = catalog.area_fraction(
        cfg, df, cfg.source.ra_src, cfg.source.dec_src)
    log(f"  SN aperture unmasked-area fraction ~ {frac_ok:.2f} "
        f"({n_flag} cells flagged)")

    # --- engine + fiducial SN sightline --------------------------------------
    log("building halo model and kappa engine ...")
    cosmo = cfg.cosmo
    hm = HaloModel(cfg.halo_model, cosmo, cfg.source.z_src)
    est = make_estimator(cfg.halo_model.mstar_method, cosmo)
    engine = KappaEngine(cfg, cosmo, hm, est, df)

    r_ap = cfg.los.aperture_radius_arcmin * 60.0
    r_in = cfg.deflector.r_exclude_arcsec
    excl_nogroup = (df["excl_deflector"] | df["is_group"]).to_numpy()
    excl_group = df["excl_deflector"].to_numpy()

    idx, theta, kappa_i = engine.kappa_los(
        cfg.source.ra_src, cfg.source.dec_src, r_in, r_ap,
        exclude_mask=excl_nogroup)
    kappa_raw_sn = float(kappa_i.sum())
    idx_g, theta_g, kappa_i_g = engine.kappa_los(
        cfg.source.ra_src, cfg.source.dec_src, r_in, r_ap,
        exclude_mask=excl_group)
    kappa_raw_sn_group = float(kappa_i_g.sum())
    n_group_members = int(df["is_group"].sum())
    log(f"  fiducial kappa_raw(SN) = {kappa_raw_sn:.4f} (group excluded), "
        f"{kappa_raw_sn_group:.4f} (group included; "
        f"{n_group_members} members)")

    # --- randoms --------------------------------------------------------------
    log(f"running {cfg.randoms.n_random_los} random sightlines ...")
    rand = randoms.run_randoms(cfg, engine, rng, progress=log)
    log(f"  kappa_random: mean {rand['kappa_mean']:.4f}, "
        f"median {rand['kappa_median']:.4f}, std {rand['kappa_std']:.4f}, "
        f"robust sigma {rand['kappa_sigma_robust']:.4f} "
        f"({rand['n_flagged']} flagged)")

    zeta = randoms.zeta_summary(cfg, engine, rand, excl_nogroup)

    # --- Monte Carlo ----------------------------------------------------------
    log(f"Monte Carlo: {cfg.montecarlo.n_mc} joint draws (group excluded) ...")
    draws_nogroup = montecarlo.mc_kappa_raw(cfg, engine, rng, excl_nogroup)
    log("Monte Carlo: group-included branch ...")
    draws_group = montecarlo.mc_kappa_raw(cfg, engine, rng, excl_group)

    k_ext_nogroup = montecarlo.build_pkappa(cfg, draws_nogroup, rand, rng)
    k_ext_group = montecarlo.build_pkappa(cfg, draws_group, rand, rng)

    # --- summary + outputs ------------------------------------------------------
    summary = {
        "target": cfg.source.name,
        "results_lens_group_excluded": corrections.corrections_summary(
            k_ext_nogroup),
        "results_lens_group_included": corrections.corrections_summary(
            k_ext_group),
        "fiducial": {
            "kappa_raw_sn_group_excluded": kappa_raw_sn,
            "kappa_raw_sn_group_included": kappa_raw_sn_group,
            "kappa_random_mean": rand["kappa_mean"],
            "kappa_random_median": rand["kappa_median"],
            "kappa_random_std": rand["kappa_std"],
            "kappa_random_sigma_robust": rand["kappa_sigma_robust"],
            "sn_percentile_among_randoms": float(
                100.0 * np.mean(rand["kappa_raw"][rand["ok"]] < kappa_raw_sn)),
            "kappa_ext_fiducial_group_excluded":
                kappa_raw_sn - rand["kappa_mean"],
            "n_group_members_excluded": n_group_members,
        },
        "zeta_number_counts": zeta,
        "catalog": {
            "n_galaxies_region": len(df),
            "n_spec": n_spec, "n_phot": n_phot,
            "n_dropped_no_z": df.attrs["n_dropped_no_z"],
            "n_in_sn_aperture": int(idx.size),
            "sn_aperture_area_fraction": frac_ok,
            "n_randoms_flagged": rand["n_flagged"],
        },
        "provenance": outputs.provenance(cfg, tap, report),
        "config": cfg.as_dict(),
    }
    path = outputs.write_summary(outdir, summary)

    # per-galaxy catalog for the SN aperture (multi-plane-ready)
    sn_cat = df.iloc[idx_g].copy()
    sn_cat["theta_arcsec"] = theta_g
    sn_cat["kappa_i"] = kappa_i_g
    sn_cat["in_default_branch"] = ~df["is_group"].to_numpy()[idx_g]
    outputs.write_catalog(outdir, sn_cat)
    outputs.write_samples(outdir, {
        "kappa_ext_group_excluded": k_ext_nogroup,
        "kappa_ext_group_included": k_ext_group,
        "kappa_raw_draws_group_excluded": draws_nogroup,
        "kappa_random": rand["kappa_raw"],
    })

    log("plotting ...")
    plots.sky_map(outdir, cfg, idx_g, theta_g, kappa_i_g, df)
    plots.cumulative_profile(outdir, cfg, theta, kappa_i)
    plots.pkappa_hist(outdir, k_ext_nogroup, k_ext_group)
    plots.randoms_hist(outdir, rand, kappa_raw_sn)
    plots.zeta_plot(outdir, cfg, rand, zeta)

    res = summary["results_lens_group_excluded"]
    k = res["kappa_ext"]
    log(f"DONE. kappa_ext = {k['p50']:+.4f} "
        f"[{k['p16']:+.4f}, {k['p84']:+.4f}] (68%), "
        f"[{k['p2.5']:+.4f}, {k['p97.5']:+.4f}] (95%)")
    log(f"  flux correction (1-k)^2 = "
        f"{res['flux_correction_(1-k)^2']['p50']:.4f}, "
        f"H0 scaling (1-k) = {res['H0_scaling_(1-k)']['p50']:.4f}")
    log(f"  summary: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
