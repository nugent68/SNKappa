#!/usr/bin/env python
"""Batch FrankenBlast SBI++ stellar-mass fits for an SNKappa target list.

Run from the frankenblast-host checkout directory, in its environment:

    cd frankenblast-host
    .venv-fb/bin/python ../SNKappa/scripts/fb_fit.py \
        --targets ../SNKappa/output/sn2025wny/fb_targets.csv \
        --out     ../SNKappa/output/sn2025wny/fb_results.csv \
        --training-root sbi_models/sbi_training_sets [--limit 50]

Bypasses FrankenBlast's Transient/photometry machinery and calls
sbi_pp.sbi_pp() directly with our Legacy Surveys photometry (already
MW-dereddened by snkappa; no further extinction correction applied here).
The trained network and training arrays are loaded ONCE (fit_host_sed's
fit_sbi_pp reloads them per object). Surviving stellar mass is
chain['logmass'] + log10(mfrac) with mfrac from a reduced number of FSPS
model.predict() calls (their pipeline uses 2500; mfrac is nearly constant
across the posterior, so ~40 samples suffice for <0.01 dex error).

Target CSV columns: ls_id, z, z_source, maggies_<F>, maggies_unc_<F> for
F in DES_g, DES_r, DES_z, WISE_W1, WISE_W2 (NaN = missing).
Appends to --out as it goes; reruns skip already-fitted ls_ids.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# chain columns for the GPD2W models, per postprocess_sbi.build_model_nonparam
# fit_order (logsfr_ratios has N=6)
THETA_LABELS = (
    ["zred", "logmass", "logzsol"]
    + [f"logsfr_ratios_{i}" for i in range(1, 7)]
    + ["dust2", "dust_index", "dust1_fraction", "log_fagn", "log_agn_tau",
       "duste_qpah", "duste_umin", "log_duste_gamma"]
)


def maggies_to_asinh(x):
    a = 2.50 * np.log10(np.e)
    mu = 35.0
    return -a * np.arcsinh((x / 2.0) * np.exp(mu / a)) + mu


def load_training(training_root, fname):
    import pickle
    out = []
    for stem in (f"hatp_x_y_{fname}_global", f"y_train_{fname}_global",
                 f"x_train_{fname}_global"):
        with open(Path(training_root) / f"{stem}.pkl", "rb") as fh:
            out.append(pickle.load(fh))
    return out


def build_noise_models(all_filters):
    from scipy.interpolate import interp1d
    meds, stds = [], []
    for f in all_filters:
        x, y = np.loadtxt(f"data/SBI/snrfiles/{f.name}_magvsnr.txt",
                          dtype=float, unpack=True)
        itp = interp1d(x, 1.0857 / y, kind="slinear",
                       fill_value="extrapolate")
        meds.append(itp)
        stds.append(itp)
    return meds, stds


def make_obs(row, all_filters):
    mags, uncs, names, waves = [], [], [], []
    for f in all_filters:
        m = row.get(f"maggies_{f.name}", np.nan)
        e = row.get(f"maggies_unc_{f.name}", np.nan)
        if np.isfinite(m) and m > 0 and np.isfinite(e):
            mags.append(maggies_to_asinh(m))
            uncs.append(2.5 / np.log(10) * e / m)
        else:
            mags.append(np.nan)
            uncs.append(np.nan)
        names.append(f.name)
        waves.append(f.wavelength_eff_angstrom)
    return {
        "mags": np.array(mags),
        "mags_unc": np.array(uncs),
        "redshift": float(row["z"]),
        "wavelengths": np.array(waves),
        "filternames": np.array(names),
    }


def mfrac_from_chain(chain, z, n_samp, model, sps, obs_for_predict, rng):
    fracs = []
    idx = rng.choice(chain.shape[0], size=min(n_samp, chain.shape[0]),
                     replace=False)
    for i in idx:
        theta = chain[i].copy()
        theta[0] = z
        try:
            _, _, mfrac = model.predict(theta, obs=obs_for_predict, sps=sps)
            fracs.append(float(mfrac))
        except Exception:
            continue
    return float(np.median(fracs)) if fracs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--training-root", required=True,
                    help="dir with hatp_x_y_*.pkl / *_train_*.pkl")
    ap.add_argument("--fname", default="zfix_GPD2W")
    ap.add_argument("--nmfrac", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())  # frankenblast-host modules
    import inspect
    import types

    import yaml
    from classes import Filter
    import sbi_pp
    # stub the imaging/dust modules postprocess_sbi pulls in at import time
    # (astroquery etc.) -- unused by the model-building path we need
    sys.modules.setdefault("mwebv_host",
                           types.SimpleNamespace(get_mwebv=None))
    sys.modules.setdefault("get_host_images",
                           types.SimpleNamespace(survey_list=None))
    from postprocess_sbi import build_model_nonparam
    from prospect.sources import FastStepBasis
    from prospect.utils.obsutils import fix_obs
    from sedpy.observate import load_filters

    # fit_host_sed.run_params (copied; importing fit_host_sed pulls the same
    # heavy import chain)
    run_params = {
        "nmc": 50, "nposterior": 50, "np_baseline": 2500, "ini_chi2": 5,
        "max_chi2": 5000, "noisy_sig": 10, "tmax_per_obj": 120000,
        "tmax_all": 600000, "outdir": "output", "verbose": False,
        "tmax_per_iter": 60,
    }

    # replicate get_host_images.survey_list() (avoids its astroquery import):
    # register Filter objects in metadata-YAML order = training-vector order
    meta = yaml.safe_load(open("data/survey_frankenblast_metadata.yml"))
    accepted = set(inspect.signature(Filter.__init__).parameters)
    for name, info in meta.items():
        kwargs = {k: v for k, v in info.items() if k in accepted}
        Filter(name=name, survey=name.split("_")[0], **kwargs)
    all_filters = Filter.all()
    print(f"filter vector ({len(all_filters)}):",
          [f.name for f in all_filters])

    print("loading training products (one-time) ...")
    hatp, y_train, x_train = load_training(args.training_root, args.fname)
    meds, stds = build_noise_models(all_filters)
    sbi_params = {
        "hatp_x_y": hatp, "y_train": y_train, "theta_train": x_train,
        "toynoise_meds_sigs": meds, "toynoise_stds_sigs": stds,
        "nhidden": 500, "nblocks": 15,
    }

    targets = pd.read_csv(args.targets)
    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        done = set(pd.read_csv(out_path)["ls_id"].astype(np.int64))
        print(f"resuming: {len(done)} already fitted")
    if args.limit:
        targets = targets.head(args.limit)

    sps = None
    model_cache_z = None, None  # (z rounded, model)
    rng = np.random.default_rng(4242)
    rows = []
    t_start = time.time()
    for n, (_, row) in enumerate(targets.iterrows()):
        lsid = np.int64(row["ls_id"])
        if lsid in done:
            continue
        obs = make_obs(row, all_filters)
        nbands = int(np.isfinite(obs["mags"]).sum())
        t0 = time.time()
        try:
            chain, obs_out, flags = sbi_pp.sbi_pp(
                obs=obs, run_params=run_params, sbi_params=sbi_params)
        except Exception as exc:
            print(f"  {lsid}: SBI failed: {exc}")
            continue
        t_sbi = time.time() - t0

        if chain is None or np.ndim(chain) != 2 or chain.shape[0] < 10:
            print(f"  {lsid}: empty chain (flags={flags})")
            continue

        # ---- surviving-mass postprocess (FSPS, lazily initialized) --------
        if sps is None:
            print("initializing FSPS stellar populations (one-time) ...")
            sps = FastStepBasis(zcontinuous=2, compute_vega_mags=False)
        z = float(row["z"])
        pobs = fix_obs({
            "wavelength": None, "spectrum": None, "unc": None,
            "redshift": z,
            "maggies": np.ones(2), "maggies_unc": np.ones(2),
            "filters": load_filters(["decam_g", "decam_z"]),
        })
        model = build_model_nonparam(pobs)
        t1 = time.time()
        mfrac = mfrac_from_chain(chain, z, args.nmfrac, model, sps, pobs, rng)
        t_mfrac = time.time() - t1

        logm = chain[:, THETA_LABELS.index("logmass")] + np.log10(mfrac)
        p16, p50, p84 = np.percentile(logm, [16, 50, 84])
        rows.append({
            "ls_id": lsid, "z": z, "z_source": row.get("z_source", ""),
            "logms_fb": p50, "logms_fb_lo": p16, "logms_fb_hi": p84,
            "logms_fb_err": 0.5 * (p84 - p16),
            "mfrac": mfrac, "nbands": nbands,
            "flag_missing": int(flags.get("use_res_missing", 0)),
            "flag_noisy": int(flags.get("use_res_noisy", 0)),
            "t_sbi_s": round(t_sbi, 1), "t_mfrac_s": round(t_mfrac, 1),
        })
        print(f"[{n+1}/{len(targets)}] {lsid}: logM*={p50:.2f} "
              f"(+{p84-p50:.2f}/-{p50-p16:.2f}) mfrac={mfrac:.3f} "
              f"nbands={nbands} sbi={t_sbi:.0f}s mfrac_t={t_mfrac:.0f}s")
        # checkpoint every object
        pd.DataFrame(rows).to_csv(
            out_path, mode="a", header=not out_path.exists(), index=False)
        rows = []

    dt = time.time() - t_start
    print(f"done in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
