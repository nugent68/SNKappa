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
    parser.add_argument("command", choices=["check", "run", "fb-export"],
                        help="check: STEP-0 availability report only; "
                             "run: full pipeline; fb-export: write the "
                             "FrankenBlast target list (see scripts/fb_fit.py)")
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
    if args.command == "fb-export":
        return fb_export(cfg, tap)
    return run_pipeline(cfg, tap, report)


def _build_catalog_engine(cfg, tap, log, overrides=None):
    """Shared by run and fb-export: fetch + clean + engine."""
    from . import catalog
    from .halos import HaloModel
    from .kappa import KappaEngine
    from .stellar import make_estimator

    df_raw, zpix = catalog.fetch_regional(cfg, tap)
    log(f"  tractor rows: {len(df_raw)}, zpix rows: {len(zpix)}")
    df = catalog.clean_and_merge(cfg, df_raw, zpix)
    cosmo = cfg.cosmo
    hm = HaloModel(cfg.halo_model, cosmo, cfg.source.z_src)
    est = make_estimator(cfg.halo_model.mstar_method, cosmo)
    kw = overrides or {}
    engine = KappaEngine(cfg, cosmo, hm, est, df, **kw)
    return df, engine, est


def fb_export(cfg, tap) -> int:
    """Write fb_targets.csv: top kappa contributors + calibration sample."""
    import time
    from pathlib import Path

    import numpy as np

    from . import fbhybrid

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    rng = np.random.default_rng(cfg.montecarlo.seed)
    outdir = Path(cfg.output.dir)
    outdir.mkdir(parents=True, exist_ok=True)
    log("building catalog and fiducial engine ...")
    df, engine, _ = _build_catalog_engine(cfg, tap, log)
    excl = (df["excl_deflector"] | df["is_group"]).to_numpy()
    idx, _, kappa_i = engine.kappa_los(
        cfg.source.ra_src, cfg.source.dec_src,
        cfg.deflector.r_exclude_arcsec,
        cfg.los.aperture_radius_arcmin * 60.0, exclude_mask=excl)
    path = fbhybrid.export_targets(cfg, df, idx, kappa_i, outdir, rng)
    log(f"wrote {path}")
    log("next: run scripts/fb_fit.py in the frankenblast-host environment, "
        "then set frankenblast.results_path in the config and rerun "
        "`snkappa run`.")
    return 0


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

    # FrankenBlast hybrid masses (if results are available)
    fb_calib = None
    engine_kwargs = {}
    fb_path = cfg.frankenblast.results_path
    if fb_path and Path(fb_path).exists():
        from . import fbhybrid
        override, err, fb_calib = fbhybrid.load_and_calibrate(cfg, df, est)
        engine_kwargs = dict(
            logms_override=override, logms_err=err,
            calib_offset=fb_calib["offset_cheap_minus_fb_dex"],
            mstar_scatter=fb_calib["scatter_dex"])
        log(f"  FrankenBlast hybrid: {fb_calib['n_fitted']} galaxies fitted; "
            f"cheap-estimator offset {fb_calib['offset_cheap_minus_fb_dex']:+.3f} dex "
            f"(subtracted everywhere), scatter {fb_calib['scatter_dex']:.3f} dex")
    elif fb_path:
        log(f"  WARNING: frankenblast.results_path set but not found: {fb_path}")

    engine = KappaEngine(cfg, cosmo, hm, est, df, **engine_kwargs)

    # --- cluster-halo tier ----------------------------------------------------
    cluster_field = None
    cluster_summary = None
    drop_members = np.zeros(len(df), dtype=bool)
    if cfg.clusters.enabled:
        from . import clusters as clu
        from .datalab import TapClient

        log("fetching cluster catalog and building cluster tier ...")
        cl_all = clu.fetch_clusters(
            cfg, lambda url: TapClient(url, cfg.data.cache_dir))
        hosts, field_cl = clu.split_host(cfg, cl_all)
        drop_members = clu.assign_members(cfg, df, cl_all, hm)
        cluster_field = clu.ClusterKappa(cfg, hm, cosmo, field_cl)
        log(f"  {len(cl_all)} clusters in region ({len(hosts)} host, "
            f"{len(field_cl)} field); {int(drop_members.sum())} member "
            f"galaxies replaced by cluster halos")

        cluster_summary = {"n_clusters_region": int(len(cl_all)),
                           "n_members_replaced": int(drop_members.sum()),
                           "hosts": []}
        for _, h in hosts.iterrows():
            hk = clu.ClusterKappa(cfg, hm, cosmo, h.to_frame().T)
            k_cen = hk.kappa_sum(cfg.source.ra_src, cfg.source.dec_src)
            k_mc = hk.mc_kappa_sum(cfg.source.ra_src, cfg.source.dec_src,
                                   rng, cfg.montecarlo.n_mc)
            dyn = clu.velocity_dispersion(cfg, df, float(h["ra"]),
                                          float(h["dec"]), float(h["z"]))
            cluster_summary["hosts"].append({
                "name": str(h["name"]), "z": float(h["z"]),
                "m200_catalog": float(h["m200"]),
                "sigma_v_diagnostic": dyn,
                "kappa_at_sn_catalog_center": k_cen,
                "kappa_at_sn_marginalized": montecarlo.percentiles(k_mc),
                "note": ("HOST cluster: degenerate with the strong-lens "
                         "model / mass-sheet; NOT included in kappa_ext"),
            })
            log(f"  host {h['name']}: kappa(SN|centered)={k_cen:.3f}; "
                f"marginalized p50={np.percentile(k_mc,50):.4f}; "
                f"sigma_v={dyn['sigma_v_kms']} km/s "
                f"(N={dyn['n_members']}) -> "
                f"M200_dyn={dyn['m200_dyn']:.2e}")

    r_ap = cfg.los.aperture_radius_arcmin * 60.0
    r_in = cfg.deflector.r_exclude_arcsec
    excl_nogroup = (df["excl_deflector"] | df["is_group"]).to_numpy() \
        | drop_members
    excl_group = df["excl_deflector"].to_numpy() | drop_members

    idx, theta, kappa_i = engine.kappa_los(
        cfg.source.ra_src, cfg.source.dec_src, r_in, r_ap,
        exclude_mask=excl_nogroup)
    kappa_cl_sn = (cluster_field.kappa_sum(cfg.source.ra_src,
                                           cfg.source.dec_src)
                   if cluster_field else 0.0)
    kappa_raw_sn = float(kappa_i.sum()) + kappa_cl_sn
    idx_g, theta_g, kappa_i_g = engine.kappa_los(
        cfg.source.ra_src, cfg.source.dec_src, r_in, r_ap,
        exclude_mask=excl_group)
    kappa_raw_sn_group = float(kappa_i_g.sum()) + kappa_cl_sn
    n_group_members = int(df["is_group"].sum())
    log(f"  fiducial kappa_raw(SN) = {kappa_raw_sn:.4f} (group excluded), "
        f"{kappa_raw_sn_group:.4f} (group included; "
        f"{n_group_members} members)"
        + (f"; field-cluster term {kappa_cl_sn:.4f}" if cluster_field
           else ""))

    # --- randoms --------------------------------------------------------------
    log(f"running {cfg.randoms.n_random_los} random sightlines ...")
    rand = randoms.run_randoms(cfg, engine, rng, progress=log,
                               base_exclude=(drop_members
                                             if cfg.clusters.enabled
                                             else None),
                               cluster_field=cluster_field)
    log(f"  kappa_random: mean {rand['kappa_mean']:.4f}, "
        f"median {rand['kappa_median']:.4f}, std {rand['kappa_std']:.4f}, "
        f"robust sigma {rand['kappa_sigma_robust']:.4f} "
        f"({rand['n_flagged']} flagged)")

    # zeta is a halo-model-independent NUMBER-count diagnostic: cluster
    # members remain real galaxies and must be counted (only their halos
    # were replaced), so use the pre-replacement exclusion mask here
    excl_counts = (df["excl_deflector"] | df["is_group"]).to_numpy()
    zeta = randoms.zeta_summary(cfg, engine, rand, excl_counts)

    # --- Monte Carlo ----------------------------------------------------------
    log(f"Monte Carlo: {cfg.montecarlo.n_mc} joint draws (group excluded) ...")
    draws_nogroup = montecarlo.mc_kappa_raw(cfg, engine, rng, excl_nogroup)
    log("Monte Carlo: group-included branch ...")
    draws_group = montecarlo.mc_kappa_raw(cfg, engine, rng, excl_group)
    if cluster_field is not None:
        log("Monte Carlo: field-cluster term (mass scatter + miscentering) ...")
        cl_draws = cluster_field.mc_kappa_sum(
            cfg.source.ra_src, cfg.source.dec_src, rng, cfg.montecarlo.n_mc)
        draws_nogroup = draws_nogroup + cl_draws
        draws_group = draws_group + cl_draws
        cluster_summary["field_kappa_at_sn"] = {
            "fiducial": kappa_cl_sn,
            "marginalized": montecarlo.percentiles(cl_draws)}

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
        "cluster_term": cluster_summary,
        "frankenblast_calibration": fb_calib,
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
