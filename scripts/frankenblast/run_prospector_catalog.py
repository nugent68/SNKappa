#!/usr/bin/env python
"""Direct prospector-alpha fits (dynesty) for SNKappa kappa contributors.

Bypasses the SBI emulator entirely — no training distribution, so the
OOD churn that blocks bright BCG-scale targets cannot occur. Uses the
SAME model the SBI was trained to emulate (fit_host_sed.build_model:
prospector-alpha, 7-bin nonparametric SFH, FastStepBasis) and the same
8-band observation builder as run_kappa_catalog (DES g/r/z + VISTA
J/H/Ks via 2MASS curves + WISE W1/W2, native errors), so results are
directly comparable to the SBI completions.

zred is left free (FastUniform 0-1.5) as in the SBI zfree model; for
spec-z targets the recovered zred_p50 vs z_best acts as a per-galaxy
quality flag.

Env knobs: PROS_NLIVE (default 200), PROS_MAXCALL (default 400000),
PROS_DLOGZ (default 0.1).

Usage: python run_prospector_catalog.py <csv> <start> <end> <summary_csv>
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

import run_kappa_catalog  # noqa: F401  (installs the 8-band BANDMAP)
import run_lens_catalog as base
import fit_host_sed
from prospect.fitting import fit_model, lnprobfn

SED_OUTPUT_ROOT = os.environ["SED_OUTPUT_ROOT"]

NLIVE = int(os.environ.get("PROS_NLIVE", "200"))
MAXCALL = int(os.environ.get("PROS_MAXCALL", "400000"))
DLOGZ = float(os.environ.get("PROS_DLOGZ", "0.1"))
FIXZ = os.environ.get("PROS_FIXZ", "1") not in ("0", "", "false")

FIT_KW = dict(optimize=False, emcee=False, dynesty=True,
              nested_method="rwalk", nlive_init=NLIVE,
              nested_dlogz_init=DLOGZ, nested_maxcall=MAXCALL,
              nested_maxcall_init=MAXCALL)


def wquant(x, w, qs):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    c /= c[-1]
    return np.interp(qs, c, x[i])


def main():
    csv_path, start, end, summary_csv = (
        sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    df = pd.read_csv(csv_path).iloc[start:end]
    for _, row in df.iterrows():
        ls_id = int(row["ls_id"])
        t0 = time.time()
        rec = {"ls_id": ls_id, "status": "ok", "n_filters": 0,
               "runtime_s": 0.0, "ncall": 0,
               "z_best": row.get("z_best"), "z_is_spec": row.get("z_is_spec")}
        try:
            obs = base.build_observations(row)
            rec["n_filters"] = len(obs["filternames"])
            comps = fit_host_sed.build_model(obs)
            model, sps = comps["model"], comps["sps"]
            # PIN THE REDSHIFT (PROS_FIXZ=1, default): the z-free
            # 8-band fits are badly degenerate -- a spec-z target came
            # back at z=0.377 against a true 0.225, dragging logM* with
            # it, and the extra dimension ate the sampling budget
            # (2 of 3 checked fits hit maxcall before dlogz < 0.1).
            # SNKappa already knows every galaxy's redshift (DESI
            # spec-z, else LS photo-z), so freeing zred throws away
            # information we have. zred is fixed by rebuilding the
            # model from its config dict (mutating free_params on a
            # live model is a no-op -- verified).
            zv = row.get("z_best", np.nan)
            if FIXZ and np.isfinite(zv):
                cfg = model.config_dict
                cfg["zred"]["isfree"] = False
                cfg["zred"]["init"] = float(zv)
                model = type(model)(cfg)
                rec["z_fixed_at"] = float(zv)
            out = fit_model(obs, model, sps, lnprobfn=lnprobfn, **FIT_KW)
            res = out["sampling"][0]
            samples = np.asarray(res.samples)
            logwt = np.asarray(res.logwt)
            w = np.exp(logwt - logwt.max())
            w /= w.sum()
            rec["ncall"] = int(np.sum(res.ncall))
            labels = model.theta_labels()
            for name in ("zred", "logmass", "logzsol", "dust2"):
                if name in labels:
                    j = labels.index(name)
                    p16, p50, p84 = wquant(samples[:, j], w,
                                           [0.16, 0.50, 0.84])
                    rec[f"{name}_p16"] = p16
                    rec[f"{name}_p50"] = p50
                    rec[f"{name}_p84"] = p84
            np.savez_compressed(
                os.path.join(SED_OUTPUT_ROOT, f"prospector_ls{ls_id}.npz"),
                samples=samples, logwt=logwt, labels=np.array(labels))
        except Exception as exc:
            rec["status"] = f"fail: {exc}"
            traceback.print_exc()
        rec["runtime_s"] = round(time.time() - t0, 1)
        pd.DataFrame([rec]).to_csv(
            summary_csv, mode="a",
            header=not os.path.exists(summary_csv), index=False)
        print(f"[{ls_id}] {rec['status']} ({rec['runtime_s']}s, "
              f"ncall {rec['ncall']})", flush=True)


if __name__ == "__main__":
    main()
