#!/usr/bin/env python
"""Absolute one-halo closure test against measured galaxy-galaxy lensing
(Phase 4 of docs/MICE_PLAN.md).

Predicts the stacked excess surface density DeltaSigma(rp) for the DESI DR1
lens samples of the "Lensing Without Borders" measurements (Heydenreich et
al. 2025, arXiv:2506.21677) using the EXACT halo chain of the SN analysis:

    dereddened LS fluxes (z, W1; served inside the DESI LSS catalogs)
    -> Nir1um stellar mass at the DESI spec-z
    -> HMF-weighted posterior <Mh|M*> (Behroozi+13, Despali+16 prior)
    -> capped at 10^13.8 Msun -> Diemer & Joyce c(M,z) -> BMO profile,

plus the stellar point-mass term, and compares with the measured
DeltaSigma per lens bin and source survey. The amplitude ratio

    A_DS = sum w ds_meas ds_pred / sum w ds_pred^2   (0.1 < rp < 1 Mpc)

is the resolution-free, data-driven test of the halo-model amplitude that
no current lightcone can provide (their lensing maps smooth exactly these
scales).

Conventions matched to the measurement pipeline (desi-lensing / dsigma):
Planck18 cosmology, comoving transverse separations, comoving DeltaSigma
(= physical / (1+z_l)^2), Msun/pc^2.

Known approximations (stated in the paper text when quoted):
- every lens is modeled as the central of its own halo (the identical
  assumption the SN pipeline makes); satellites boost the measured signal
  at rp ~ 0.5-1 Mpc, so A_DS > 1 there partially reflects satellites, not
  a low model amplitude. The rp < 0.3 Mpc bins are the cleanest.
- lens weighting uses the LSS catalog WEIGHT (the measurement additionally
  weights by Sigma_crit^-2 per lens-source pair; the bins are narrow in z).
- the 2-halo term is not modeled; the fit range stops at 1 Mpc.

Run on Perlmutter:  ./venv/bin/python scripts/delta_sigma_closure.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.config import HaloModelConfig
from snkappa.halos import HaloModel, bmo_table
from snkappa.stellar import make_estimator

BASE = Path("/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing")
CATDIR = BASE / "desi_catalogues/v1.5"
MEASDIR = BASE / "lensing_measurements/v1.5"
SURVEYS = ("DES", "KiDS", "HSCY3", "SDSS")

# lens bins of arXiv:2506.21677: (sample, zmin, zmax, M_R cut or None)
BINS = [
    ("BGS_BRIGHT", 0.10, 0.20, -19.5),
    ("BGS_BRIGHT", 0.20, 0.30, -20.5),
    ("BGS_BRIGHT", 0.30, 0.40, -21.0),
    ("LRG", 0.40, 0.60, None),
    ("LRG", 0.60, 0.80, None),
    ("LRG", 0.80, 1.10, None),
]
RP_FIT = (0.10, 1.00)   # comoving Mpc


def _mag(flux):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 22.5 - 2.5 * np.log10(np.where(flux > 0, flux, np.nan))


def mags_from_fluxes(t, sample):
    """Dereddened AB mags per band. BGS clustering catalogs serve
    flux_*_dered directly; LRG needs a TARGETID join against the full
    LSS catalog's raw fluxes + MW transmissions."""
    if "flux_z_dered" in t.colnames:
        return {b: _mag(np.asarray(t[f"flux_{b}_dered"], dtype=float))
                for b in ("g", "r", "z", "w1")}
    from astropy.io import fits
    with fits.open(CATDIR / f"{sample}_full_HPmapcut.dat.fits",
                   memmap=True) as h:
        d = h[1].data
        tid_full = d["TARGETID"]
        order = np.argsort(tid_full)
        pos = np.searchsorted(tid_full[order], np.asarray(t["TARGETID"]))
        pos = np.clip(pos, 0, order.size - 1)
        row = order[pos]
        assert (tid_full[row] == np.asarray(t["TARGETID"])).mean() > 0.999
        out = {}
        for b in ("g", "r", "z", "w1"):
            f = d[f"FLUX_{b.upper()}"][row].astype(float)
            mw = d[f"MW_TRANSMISSION_{b.upper()}"][row].astype(float)
            out[b] = _mag(np.where(mw > 0, f / np.where(mw > 0, mw, 1), -1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=30000,
                    help="max lenses per bin (uniform subsample)")
    ap.add_argument("--smhm-inverse", default="posterior",
                    choices=("posterior", "naive"))
    ap.add_argument("--logmh-max", type=float, default=13.8)
    ap.add_argument("--mstar-method", default="nir1um")
    ap.add_argument("--variant", default="")
    ap.add_argument("--out", default="output/delta_sigma")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20130901)

    from astropy.cosmology import Planck18 as cosmo

    class _CosmoCfg:                     # HaloModel wants .H0/.Om0 access
        pass
    hcfg = HaloModelConfig(smhm_inverse=args.smhm_inverse,
                           logmh_max=args.logmh_max)
    hm = HaloModel(hcfg, cosmo, 1.2)
    est = make_estimator(args.mstar_method, cosmo)
    tab = bmo_table()

    results = {"args": vars(args), "bins": []}
    for sample, zlo, zhi, mcut in BINS:
        t = Table.read(CATDIR / f"{sample}_clustering.dat.fits")
        sel = (np.asarray(t["Z"]) > zlo) & (np.asarray(t["Z"]) < zhi)
        absmag_col = "ABSMAG_RP1" if "ABSMAG_RP1" in t.colnames else "ABSMAG_R"
        if mcut is not None:
            sel &= np.asarray(t[absmag_col], dtype=float) < mcut
        idx = np.flatnonzero(sel)
        n_bin = idx.size
        if idx.size > args.n_max:
            idx = rng.choice(idx, args.n_max, replace=False)
        t = t[idx]
        z = np.asarray(t["Z"], dtype=float)
        w = np.asarray(t["WEIGHT"], dtype=float)
        mags = mags_from_fluxes(t, sample)
        logms = est.logmstar(mags, z)
        ok = np.isfinite(logms)
        z, w, logms = z[ok], w[ok], logms[ok]

        ib = hm.zbin_index(z)
        rhos, rs, tau = hm.halo_params(logms, ib)

        # measured rp grid from the first available survey file
        meas = {}
        for sv in SURVEYS:
            for boost in ("True", "False"):
                f = (MEASDIR / sv / f"deltasigma_{sample}_zmin_{zlo}_zmax_"
                     f"{zhi}_blindA_boost_{boost}.fits")
                if f.exists():
                    m = Table.read(f)
                    meas[sv] = {"rp": np.asarray(m["rp"], float),
                                "ds": np.asarray(m["ds"], float),
                                "err": np.asarray(m["ds_err"], float),
                                "boost": boost}
                    break
        if not meas:
            print(f"!! no measurements found for {sample} {zlo}-{zhi}")
            continue
        rp = next(iter(meas.values()))["rp"]          # comoving Mpc

        # stacked prediction on that grid: comoving rp -> physical R at
        # each lens; comoving DeltaSigma = physical / (1+z)^2
        r_phys = rp[None, :] / (1.0 + z[:, None])     # [n_lens, n_rp] Mpc
        x = r_phys / rs[:, None]
        ds_h = (rhos * rs)[:, None] * tab.delta_sigma_dimless(
            x, np.broadcast_to(tau[:, None], x.shape))
        ds_star = 10.0 ** logms[:, None] / (np.pi * (r_phys * 1e6) ** 2)
        ds_phys = ds_h / 1e12 + ds_star               # Msun / pc^2
        ds_com = ds_phys / (1.0 + z[:, None]) ** 2
        pred = np.average(ds_com, axis=0, weights=w)

        r200_phys = rs * tau                       # truncation = r200c [Mpc]
        r200_med = float(np.median(r200_phys))
        zl_med = float(np.median(z))
        entry = {"sample": sample, "zmin": zlo, "zmax": zhi,
                 "absmag_col": absmag_col if mcut else None,
                 "mcut": mcut, "n_lenses_bin": int(n_bin),
                 "n_used": int(z.size),
                 "logms_med": float(np.median(logms)),
                 "logmh_med": float(np.median(np.log10(
                     (4/3) * np.pi * 200 * hm.rhoc[ib] * r200_phys**3))),
                 "r200_med_phys": r200_med,
                 "rp": rp.tolist(), "ds_pred": pred.tolist(),
                 "surveys": {}}
        # fit windows [comoving Mpc]: the 1-halo-dominated window scales
        # with the halo size; beyond ~2 r200 the measurement picks up
        # NEIGHBORING halos, which the SN pipeline's catalog sum includes
        # but this single-halo stack deliberately does not.
        r1h = 2.0 * r200_med * (1.0 + zl_med)      # comoving
        windows = {"one_halo": (0.08, max(0.25, r1h)),
                   "fiducial": RP_FIT, "neighbor": (1.0, 3.0)}
        for sv, m in meas.items():
            svent = {"boost": m["boost"], "ds_meas": m["ds"].tolist(),
                     "ds_err": m["err"].tolist(), "windows": {}}
            for wname, (rlo, rhi) in windows.items():
                infit = (rp > rlo) & (rp < rhi)
                if infit.sum() < 2:
                    continue
                wgt = 1.0 / m["err"][infit] ** 2
                a = (np.sum(wgt * m["ds"][infit] * pred[infit])
                     / np.sum(wgt * pred[infit] ** 2))
                ea = 1.0 / np.sqrt(np.sum(wgt * pred[infit] ** 2))
                svent["windows"][wname] = {"rp_range": [rlo, rhi],
                                           "A_ds": float(a),
                                           "A_ds_err": float(ea)}
            oh = svent["windows"].get("one_halo", {})
            print(f"{sample} z[{zlo},{zhi}] {sv:6s} "
                  f"A_1h = {oh.get('A_ds', float('nan')):.3f} "
                  f"+- {oh.get('A_ds_err', float('nan')):.3f} "
                  f"(r<{windows['one_halo'][1]:.2f}) | "
                  f"logM*={entry['logms_med']:.2f} "
                  f"logMh={entry['logmh_med']:.2f} "
                  f"r200={r200_med:.2f}", flush=True)
            entry["surveys"][sv] = svent
        results["bins"].append(entry)
        suffix = f"_{args.variant}" if args.variant else ""
        (outdir / f"delta_sigma_closure{suffix}.json").write_text(
            json.dumps(results, indent=2, default=float))

    print(f"saved {outdir / f'delta_sigma_closure{suffix}.json'}")


if __name__ == "__main__":
    main()
