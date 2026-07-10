#!/usr/bin/env python
"""End-to-end mock calibration on cosmoDC2 (TODO 5.1).

Runs the IDENTICAL kappa-prediction pipeline used for DES
(snkappa.batch.BatchEngine + ClusterField, same apertures, same
source-plane binning + interpolation, same random-sightline zero point)
on cosmoDC2 mock regions where the TRUE ray-traced convergence at every
sightline is known, with DES-like observational noise forward-modeled:

- stellar masses: true log M* + 0.16 dex Gaussian (the validated estimator
  scatter), plus the estimator's distance-modulus dependence on the assumed
  redshift (MockEstimator);
- photo-z: split-normal quantiles from z_true + sigma_z0*(1+z) Gaussian,
  with a catastrophic-outlier fraction resampled uniformly on (0, 1.2]
  (rates measured against DESI in scripts/photoz_validation.py);
- no spec-z (photo-only, matching the C/E fields that carry the DES signal);
- cluster catalog: true halo mass scattered by 0.25 dex lognormal to form
  the "richness mass" (so the catalog-mass Eddington bias is present in the
  mock exactly as in Wen & Han).

The recovered amplitude A_mock = d<kappa_true>/d kappa_pred calibrates, in
one number, the combined effect of: SMHM-inversion bias under a non-Behroozi
galaxy-halo connection, the single-galaxy cap, the missing 2-halo term,
photo-z and M* noise attenuation, and the cluster-mass conventions. A second
fit adds SN-like magnitude noise to show what DES actually measures.

Cosmology is set to cosmoDC2's (flat LCDM, H0=71, Om0=0.2648). The colossus
concentration/HMF internals keep Ob0=0.048, sigma8=0.81, ns=0.96 (vs DC2's
0.0448/0.80/0.963) -- a <2% effect on c(M,z), noted for completeness.

Run (Perlmutter):
  ./venv/bin/python scripts/mock_calibration.py --pixels 10066 ... \
      --datadir mockdata --out output/mock_dc2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snkappa.config import (Config, ClustersConfig, CosmologyConfig,
                            DataConfig, DeflectorConfig, HaloModelConfig,
                            LensGroupConfig, LosConfig, MonteCarloConfig,
                            OutputConfig, RandomsConfig, SourceConfig)
from snkappa import catalog, clusters as clu
from snkappa.batch import BatchEngine, ClusterField
from snkappa.fitting import bootstrap_slope
from snkappa.halos import HaloModel

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

MAG_LIMIT = 22.5
APERTURE_ARCMIN = 10.0
R_IN = 3.0
ZSRC_EDGES = np.arange(0.10, 1.1501, 0.05)
ZSRC_CENTERS = 0.5 * (ZSRC_EDGES[:-1] + ZSRC_EDGES[1:])
ZC = np.arange(0.02, 1.14, 0.04)
SLOPE_TH = -5.0 / np.log(10.0)


class MockEstimator:
    """Returns each galaxy's noisy true logM*, with the real estimator's
    distance-modulus dependence on the ASSUMED redshift.

    Column conventions (set up in build_df): mag_g carries the per-galaxy
    noisy logM* evaluated at z_true; mag_z carries z_true. When the batch
    engine evaluates a photo-z galaxy at grid redshift z, the estimate
    shifts by 0.4*(DM(z) - DM(z_true)), exactly as a luminosity-based
    estimator would. (The K-correction's z-dependence, subdominant to the
    distance modulus, is not modeled.)
    """

    def __init__(self, cosmo):
        self.cosmo = cosmo
        zg = np.linspace(1e-3, 1.6, 400)
        self._dm = (zg, cosmo.distmod(zg).value)

    def logmstar(self, mags, z):
        lm0 = np.asarray(mags["g"], dtype=float)
        zt = np.asarray(mags["z"], dtype=float)
        z = np.asarray(z, dtype=float)
        dm = np.interp(np.clip(z, 1e-3, None), *self._dm)
        dm0 = np.interp(np.clip(zt, 1e-3, None), *self._dm)
        return np.clip(lm0 + 0.4 * (dm - dm0), 7.0, 12.2)


def make_cfg(args, center):
    return Config(
        source=SourceConfig(name="DC2", ra_src=center[0], dec_src=center[1],
                            z_src=1.15),
        deflector=DeflectorConfig(ra_lens=center[0], dec_lens=center[1],
                                  z_lens=0.3, r_exclude_arcsec=R_IN),
        lens_group=LensGroupConfig(r_group_arcmin=0.01),
        los=LosConfig(aperture_radius_arcmin=APERTURE_ARCMIN,
                      mag_limit=MAG_LIMIT),
        randoms=RandomsConfig(n_random_los=args.n_rand, annulus_deg=[0.1, 1.5]),
        montecarlo=MonteCarloConfig(n_mc=64, seed=20130901),
        data=DataConfig(),
        halo_model=HaloModelConfig(smhm_inverse=args.smhm_inverse,
                                   logmh_max=args.logmh_max),
        cosmology=CosmologyConfig(H0=71.0, Om0=0.2648),   # cosmoDC2
        clusters=ClustersConfig(enabled=True),
        output=OutputConfig(dir="output/mock_dc2"),
    )


def build_df(fg, rng, args):
    """Forward-model DES-like observational noise onto the truth catalog."""
    n = len(fg)
    zt = fg.redshift_true.to_numpy()
    sig = args.sigz0 * (1.0 + zt)
    zp = zt + rng.standard_normal(n) * sig
    out = rng.random(n) < args.outlier_frac
    zp[out] = rng.uniform(0.01, 1.2, int(out.sum()))
    zp = np.clip(zp, 0.011, None)
    lm = (np.log10(fg.stellar_mass.to_numpy())
          + rng.standard_normal(n) * args.mstar_scatter)
    return pd.DataFrame({
        "ra": fg.ra.to_numpy(), "dec": fg.dec.to_numpy(),
        "z_spec": np.nan,
        "zp_med": zp, "zp_std": sig,
        "zp_l68": zp - sig, "zp_u68": zp + sig,
        "mag_g": lm,          # MockEstimator: noisy logM* at z_true
        "mag_z": zt,          # MockEstimator: z_true (for the DM term)
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", type=int, nargs="+", required=True)
    ap.add_argument("--datadir", default="mockdata")
    ap.add_argument("--out", default="output/mock_dc2")
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--sigz0", type=float, default=0.03)
    ap.add_argument("--outlier-frac", type=float, default=0.02)
    ap.add_argument("--mstar-scatter", type=float, default=0.16)
    ap.add_argument("--clmass-scatter", type=float, default=0.25)
    ap.add_argument("--smhm-inverse", default="posterior",
                    choices=("posterior", "naive"))
    ap.add_argument("--logmh-max", type=float, default=13.8)
    ap.add_argument("--sn-noise", type=float, default=0.15,
                    help="mag noise for the DES-like fit")
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--variant", default="")
    args = ap.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    r_out = APERTURE_ARCMIN * 60.0
    rows = []

    for pix in args.pixels:
        fg = pd.read_parquet(datadir / f"fg_{pix}.parquet")
        src = pd.read_parquet(datadir / f"src_{pix}.parquet")
        clt = pd.read_parquet(datadir / f"cl_{pix}.parquet")
        center = (float(fg.ra.mean()), float(fg.dec.mean()))
        cfg = make_cfg(args, center)
        df = build_df(fg, rng, args)

        # mock "richness" masses: truth scattered by the W&H-like lognormal
        cl = pd.DataFrame({
            "name": [f"mock{pix}_{i}" for i in range(len(clt))],
            "ra": clt.ra.to_numpy(), "dec": clt.dec.to_numpy(),
            "z": clt.redshift_true.to_numpy(),
            "m200": clt.halo_mass.to_numpy()
            * 10.0 ** (rng.standard_normal(len(clt)) * args.clmass_scatter),
        })

        hm = HaloModel(cfg.halo_model, cfg.cosmo, 1.15)
        members = clu.assign_members(cfg, df, cl, hm)
        df = df[~members].reset_index(drop=True)
        est = MockEstimator(cfg.cosmo)
        eng = BatchEngine(cfg, df, hm, est, ZC, R_IN)
        clf = ClusterField(cl, hm, mass_scatter_dex=args.clmass_scatter,
                           miscenter_frac_r500=0.2, conc=5.0)
        log(f"pixel {pix}: fg {len(df)} (members {int(members.sum())}), "
            f"cl {len(cl)}, src {len(src)}")

        # interior guard for randoms and sources: the healpix diamond has no
        # rectangular footprint, so require a fully populated aperture
        def interior(ra0, dec0):
            frac, _ = catalog.area_fraction(cfg, df, ra0, dec0)
            return frac >= 0.999

        lo_ra, hi_ra = df.ra.min(), df.ra.max()
        lo_de, hi_de = df.dec.min(), df.dec.max()
        rra, rdec = [], []
        tries = 0
        while len(rra) < args.n_rand and tries < 40 * args.n_rand:
            tries += 1
            a = rng.uniform(lo_ra, hi_ra); d = rng.uniform(lo_de, hi_de)
            if eng.counts(a, d, r_out) > 0 and interior(a, d):
                rra.append(a); rdec.append(d)
        rra, rdec = np.array(rra), np.array(rdec)

        ok_src = np.array([interior(r.ra, r.dec)
                           for r in src.itertuples()])
        src = src[ok_src].reset_index(drop=True)
        log(f"  {rra.size} randoms, {len(src)} interior sources")

        zc0, dz = ZSRC_CENTERS[0], ZSRC_CENTERS[1] - ZSRC_CENTERS[0]
        pos = np.clip((src.redshift_true.to_numpy() - zc0) / dz, 0.0,
                      ZSRC_CENTERS.size - 1.0)
        k_lo = np.floor(pos).astype(int)
        k_hi = np.minimum(k_lo + 1, ZSRC_CENTERS.size - 1)
        t_int = pos - k_lo
        t_int[k_hi == k_lo] = 0.0

        kap_sn = np.full((len(src), 2), np.nan)
        zp_c = np.full(ZSRC_CENTERS.size, np.nan)
        for k, z_src in enumerate(ZSRC_CENTERS):
            need_lo = np.flatnonzero(k_lo == k)
            need_hi = np.flatnonzero((k_hi == k) & (k_lo != k))
            if need_lo.size + need_hi.size == 0:
                continue
            eng.set_zsrc(cfg.cosmo, z_src)
            clf.set_zsrc(cfg.cosmo, z_src)

            def kap(a, d):
                return eng.kappa_gal(a, d, r_out) + clf.kappa_sum(a, d)

            k_rand = np.array([kap(a, d) for a, d in zip(rra, rdec)])
            zp_c[k] = k_rand.mean()
            for i in np.concatenate([need_lo, need_hi]):
                col = 0 if k_lo[i] == k else 1
                kap_sn[i, col] = kap(src.ra.iloc[i], src.dec.iloc[i])
            log(f"  z_src={z_src:.3f}: {need_lo.size + need_hi.size:3d} "
                f"src evals | zp {zp_c[k]:.4f}")

        for i in range(len(src)):
            t = t_int[i]; lo, hi = k_lo[i], k_hi[i]
            k_raw = (1 - t) * kap_sn[i, 0] + t * (kap_sn[i, 1]
                                                  if t > 0 else 0.0)
            zp = (1 - t) * zp_c[lo] + t * (zp_c[hi] if t > 0 else 0.0)
            rows.append({
                "pixel": pix, "z": src.redshift_true.iloc[i],
                "kappa_pred": k_raw - zp,
                "kappa_true": src.convergence.iloc[i],
                "zbin": ZSRC_CENTERS[lo if t < 0.5 else hi]})

    res = pd.DataFrame(rows)
    # per (pixel, zbin) demeaning of the truth, mirroring the DES per-bin
    # Hubble-residual demeaning (kappa_pred is already zero-pointed per pixel)
    res["kt_demean"] = res.kappa_true
    for _, idx in res.groupby(["pixel", "zbin"]).groups.items():
        res.loc[idx, "kt_demean"] -= res.loc[idx, "kt_demean"].mean()
    suffix = f"_{args.variant}" if args.variant else ""
    res.to_csv(outdir / f"mock_pairs{suffix}.csv", index=False)

    x = res.kappa_pred.to_numpy()
    yk = res.kt_demean.to_numpy()
    ones = np.ones_like(x)
    A_mock, eA = bootstrap_slope(x, yk, ones, rng)      # unweighted OLS
    # DES-like fit: mu residuals with SN noise
    y_mu = SLOPE_TH * res.kappa_true.to_numpy() \
        + rng.standard_normal(len(res)) * args.sn_noise
    for _, idx in res.groupby(["pixel", "zbin"]).groups.items():
        y_mu[res.index.get_indexer(idx)] -= y_mu[
            res.index.get_indexer(idx)].mean()
    b_sn, e_sn = bootstrap_slope(x, y_mu, np.full_like(x, args.sn_noise), rng)

    r2 = float(np.corrcoef(x, yk)[0, 1] ** 2)
    summary = {
        "variant": args.variant or "fiducial", "args": vars(args),
        "n_src": len(res),
        "A_mock_kappa": [A_mock, eA],
        "A_mock_desfit": [b_sn / SLOPE_TH, abs(e_sn / SLOPE_TH)],
        "r2": r2,
        "var_pred": float(x.var()), "var_true": float(yk.var()),
        "cov": float(np.cov(x, yk)[0, 1]),
    }
    (outdir / f"mock_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, default=float))
    log("=" * 60)
    log(f"N={len(res)}  A_mock(kappa) = {A_mock:.3f} +- {eA:.3f}  r2={r2:.3f}")
    log(f"DES-like fit (sigma={args.sn_noise} mag): "
        f"A = {b_sn/SLOPE_TH:.3f} +- {abs(e_sn/SLOPE_TH):.3f}")
    log(f"sig(kappa_pred)={x.std():.4f}  sig(kappa_true)={yk.std():.4f}")
    log(f"saved {outdir / f'mock_summary{suffix}.json'}")


if __name__ == "__main__":
    main()
