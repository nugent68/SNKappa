#!/usr/bin/env python
"""DES-SN5YR X-field pilot: per-SN external convergence vs Hubble residuals.

Tests whether SNKappa-style kappa (validated masses + DESI spec-z + cluster
tier) correlates with DES Hubble residuals, a la Shah et al. 2024 but with
an independently calibrated (predicted, not residual-fitted) halo model.

Strategy: ONE regional catalog covering the adjacent X1/X2/X3 fields; SNe
binned in source redshift (the engine's Sigma_crit tables are per-z_src);
per bin, evaluate kappa at each SN host position and at random sightlines
(zero point). Cluster tier: Wen & Han region-wide with member replacement.

Run: .venv/bin/python scripts/des_pilot.py   (from the SNKappa directory)
"""

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
from snkappa.datalab import TapClient
from snkappa.halos import HaloModel
from snkappa.kappa import KappaEngine
from snkappa.stellar import make_estimator

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------- inputs --
FIELDS = ("X1", "X2", "X3")
CENTER = (35.6, -5.3)          # XMM-LSS field-cluster centroid
REGION_ANNULUS = [0.5, 2.15]   # sets fetch radius = 2.15 + aperture
APERTURE_ARCMIN = 10.0
R_INNER_ARCSEC = 3.0           # guards profile divergence + the SN host
ZBIN_W = 0.1
N_RAND_PER_BIN = 150
OMEGA_M = 0.352                # DES-SN5YR flat LCDM best fit (SN only)


def read_snana(path):
    rows, names = [], None
    for line in open(path):
        if line.startswith("VARNAMES:"):
            names = line.split()[1:]
        elif line.startswith("SN:"):
            rows.append(line.split()[1:])
    return pd.DataFrame(rows, columns=names)


def load_des():
    hd = read_snana("des_pilot/DES_HD.csv")
    meta = read_snana("des_pilot/DES_meta.csv")
    m = meta[meta.FIELD.isin(FIELDS)][
        ["CID", "FIELD", "HOST_RA", "HOST_DEC", "HOST_ZSPEC"]]
    sn = hd.merge(m, on="CID", how="inner")
    for c in ("zHD", "MU", "MUERR", "MUERR_SYS", "PROBIA_BEAMS",
              "HOST_RA", "HOST_DEC"):
        sn[c] = pd.to_numeric(sn[c], errors="coerce")
    sn = sn[np.isfinite(sn.zHD) & np.isfinite(sn.MU) & np.isfinite(sn.HOST_RA)
            & (sn.HOST_RA > -100) & (sn.zHD > 0.1) & (sn.zHD < 1.15)]
    return sn.reset_index(drop=True)


def make_cfg(z_src):
    return Config(
        source=SourceConfig(name="DESX", ra_src=CENTER[0], dec_src=CENTER[1],
                            z_src=z_src),
        deflector=DeflectorConfig(ra_lens=CENTER[0], dec_lens=CENTER[1],
                                  z_lens=0.3, r_exclude_arcsec=R_INNER_ARCSEC),
        lens_group=LensGroupConfig(r_group_arcmin=0.01),
        los=LosConfig(aperture_radius_arcmin=APERTURE_ARCMIN, mag_limit=21.0),
        randoms=RandomsConfig(n_random_los=N_RAND_PER_BIN,
                              annulus_deg=REGION_ANNULUS),
        montecarlo=MonteCarloConfig(n_mc=100, seed=20130901),
        data=DataConfig(),
        halo_model=HaloModelConfig(),
        cosmology=CosmologyConfig(),
        clusters=ClustersConfig(enabled=True),
        output=OutputConfig(dir="output/des_pilot"),
    )


def main():
    outdir = Path("output/des_pilot"); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20130901)
    sn = load_des()
    log(f"{len(sn)} X-field SNe (zHD 0.1-1.15)")

    # ---- one regional fetch at the maximum source redshift -----------------
    cfg_fetch = make_cfg(1.15)
    tap = TapClient(cfg_fetch.data.tap_url, cfg_fetch.data.cache_dir)
    df = catalog.clean_and_merge(cfg_fetch, *catalog.fetch_regional(cfg_fetch, tap))
    n_spec = int(df.z_spec.notna().sum())
    log(f"regional catalog: {len(df)} galaxies ({n_spec} DESI spec-z)")

    cl_all = clu.fetch_clusters(cfg_fetch,
                                lambda url: TapClient(url, cfg_fetch.data.cache_dir))
    hm_fetch = HaloModel(cfg_fetch.halo_model, cfg_fetch.cosmo, 1.15)
    members = clu.assign_members(cfg_fetch, df, cl_all, hm_fetch)
    log(f"{len(cl_all)} W&H clusters in region; "
        f"{int(members.sum())} member galaxies replaced")

    est = make_estimator("nir1um", cfg_fetch.cosmo)
    r_out = APERTURE_ARCMIN * 60.0

    # random sightline positions (shared across bins)
    th = np.sqrt(rng.uniform(0, 1, N_RAND_PER_BIN)) * 1.7
    ph = rng.uniform(0, 2*np.pi, N_RAND_PER_BIN)
    rand_ra = CENTER[0] + th*np.cos(ph)/np.cos(np.radians(CENTER[1]))
    rand_dec = CENTER[1] + th*np.sin(ph)

    # ---- per z-bin engines --------------------------------------------------
    zedges = np.arange(0.1, 1.2001, ZBIN_W)
    results = []
    for lo, hi in zip(zedges[:-1], zedges[1:]):
        in_bin = sn[(sn.zHD >= lo) & (sn.zHD < hi)]
        if len(in_bin) == 0:
            continue
        z_src = 0.5*(lo+hi)
        cfg = make_cfg(z_src)
        hm = HaloModel(cfg.halo_model, cfg.cosmo, z_src)
        # drop confirmed-background spec galaxies for THIS source plane
        keep = ~(df.z_spec >= z_src - 0.01)
        df_bin = df[keep].reset_index(drop=True)
        memb_bin = members[keep.to_numpy()]
        eng = KappaEngine(cfg, cfg.cosmo, hm, est, df_bin)
        cl_bin = cl_all[cl_all.z < z_src - 0.02]
        ck = clu.ClusterKappa(cfg, hm, cfg.cosmo, cl_bin)

        def kap(ra, dec):
            _, _, k = eng.kappa_los(ra, dec, R_INNER_ARCSEC, r_out,
                                    exclude_mask=memb_bin)
            return float(k.sum()) + ck.kappa_sum(ra, dec)

        k_rand = np.array([kap(r, d) for r, d in zip(rand_ra, rand_dec)])
        zp, sig = k_rand.mean(), 0.5*np.subtract(*np.percentile(k_rand, [84, 16]))
        for _, row in in_bin.iterrows():
            k = kap(row.HOST_RA, row.HOST_DEC)
            results.append({"CID": row.CID, "FIELD": row.FIELD,
                            "zHD": row.zHD, "MU": row.MU, "MUERR": row.MUERR,
                            "PROBIA": row.PROBIA_BEAMS,
                            "kappa_raw": k, "kappa_ext": k - zp,
                            "zbin": z_src, "rand_mean": zp, "rand_sig": sig})
        log(f"z bin [{lo:.2f},{hi:.2f}): {len(in_bin):3d} SNe | "
            f"rand mean {zp:.4f} robust sig {sig:.4f}")

    res = pd.DataFrame(results)

    # ---- Hubble residuals and correlation -----------------------------------
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=OMEGA_M)
    res["mu_model"] = cosmo.distmod(res.zHD.to_numpy()).value
    w = 1.0/res.MUERR**2
    res["hr"] = res.MU - res.mu_model
    res["hr"] -= np.average(res.hr, weights=w)

    x, y = res.kappa_ext.to_numpy(), res.hr.to_numpy()
    b, a = np.polyfit(x, y, 1, w=w)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(res), len(res))
        boot.append(np.polyfit(x[i], y[i], 1, w=w.to_numpy()[i])[0])
    berr = np.std(boot)
    rho = np.corrcoef(x, y)[0, 1]
    log("=" * 60)
    log(f"N = {len(res)} SNe | sigma(kappa_ext) = {x.std():.4f} | "
        f"sigma(HR) = {y.std():.3f}")
    log(f"slope dHR/dkappa = {b:+.2f} +- {berr:.2f}  (lensing predicts -2.17)")
    log(f"pearson rho = {rho:+.4f}  -> significance {abs(b)/berr:.1f} sigma")
    hi_k = res.nlargest(8, "kappa_ext")
    log("top kappa sightlines:")
    for _, r in hi_k.iterrows():
        log(f"  {r.CID} ({r.FIELD}) z={r.zHD:.3f} kappa={r.kappa_ext:+.4f} "
            f"HR={r.hr:+.3f}")
    # full-survey projection
    proj = abs(b)/berr * np.sqrt(1635/len(res))
    log(f"naive full-survey projection (x sqrt(1635/{len(res)})): "
        f"{proj:.1f} sigma IF slope holds")
    res.to_csv(outdir / "des_xfields_kappa.csv", index=False)
    log(f"saved {outdir/'des_xfields_kappa.csv'}")


if __name__ == "__main__":
    main()
