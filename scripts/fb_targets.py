#!/usr/bin/env python
"""FrankenBlast 8-band campaign: select top-kappa contributor galaxies.

For every SN in the three NIR-field kappa catalogs, evaluate each
aperture galaxy's individual convergence contribution with the SAME
tables the survey pipeline uses (BatchEngine internals: spec galaxies
kappa_i = sA * sigma_dimless(theta/theta_s, tau); photo galaxies the
p(z)-weighted sum over the marginalization grid), rank, and keep the
union of top contributors. These few-percent-of-catalog galaxies carry
most of the predicted signal, so improving THEIR masses with 8-band
FrankenBlast SBI (g,r,z + J,H,Ks + W1,W2) buys the most accuracy per
CPU hour.

Selection per SN: kappa_i >= KAPPA_MIN, or rank <= TOP_PER_SN.
FB list additionally requires >= 2 finite deep-NIR bands.

Output (gitignored): data_nir/fb_targets_phot.csv — runner-compatible
Tractor-style columns (nanomaggies + ivar + mw_transmission per band;
NIR fluxes are pre-dereddened so their mw_transmission is 1.0) plus
bookkeeping (z_best, logM*, kappa stats, survey/region provenance).

Run: .venv/bin/python scripts/fb_targets.py
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snkappa import catalog, clusters as clu
from snkappa.batch import BatchEngine
from snkappa.datalab import TapClient
from snkappa.halos import HaloModel
from snkappa.kappa import angular_sep_arcsec
from snkappa.nir import attach_nir
from snkappa.stellar import make_estimator
import des_full
import union3_full as uf

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


OUT = Path("data_nir/fb_targets_phot.csv")
KAPPA_MIN = 2e-4
TOP_PER_SN = 10
R_OUT = 600.0
R_IN = 3.0
NIR_KAPPA = {
    "des": ("output/nir_video/des_all_kappa_nir.csv", None),
    "union3": ("output/nir_video/union3/union3_kappa.csv",
               "output/nir_video/union3/regions.csv"),
    "pantheon": ("output/nir_video/pantheon/pantheon_kappa.csv",
                 "output/nir_video/pantheon/regions.csv"),
}


def contributions(eng, ra0, dec0):
    """(index_into_df, kappa_i) for galaxies contributing at (ra0, dec0)."""
    out_idx, out_k = [], []
    b = np.abs(eng.s_dec - dec0) < R_OUT / 3600.0
    th = angular_sep_arcsec(ra0, dec0, eng.s_ra[b], eng.s_dec[b])
    m = (th > R_IN) & (th < R_OUT) & (eng.sA[b] > 0)
    if m.any():
        k = eng.sA[b][m] * eng.hm.sigma_dimless(
            th[m] / eng.s_ths[b][m], eng.s_tau[b][m])
        out_idx.append(eng.s_dfidx[b][m])
        out_k.append(k)
    b = np.abs(eng.p_dec - dec0) < R_OUT / 3600.0
    th = angular_sep_arcsec(ra0, dec0, eng.p_ra[b], eng.p_dec[b])
    m = (th > R_IN) & (th < R_OUT)
    if m.any():
        ths = eng.p_ths[b][m]
        x = th[m][:, None] / ths
        sig = eng.hm.sigma_dimless(
            x.ravel(), eng.p_tau[b][m].ravel()).reshape(x.shape)
        out_idx.append(eng.p_dfidx[b][m])
        out_k.append(np.sum(eng.pWA[b][m] * sig, axis=1))
    if not out_idx:
        return np.array([], int), np.array([])
    return np.concatenate(out_idx), np.concatenate(out_k)


def tag_engine_indices(eng, df):
    """Map engine spec/phot rows back to df row positions."""
    spec = df.z_spec.notna().to_numpy()
    eng.s_dfidx = np.flatnonzero(spec)
    eng.p_dfidx = np.flatnonzero(~spec)


def run_region(cfg, df, sn, hm, est, hits, prov):
    eng = BatchEngine(cfg, df, hm, est, uf.ZC, R_IN)
    tag_engine_indices(eng, df)
    for zb, grp in sn.groupby("zbin"):
        eng.set_zsrc(cfg.cosmo, float(zb))
        for _, row in grp.iterrows():
            idx, k = contributions(eng, row.HOST_RA, row.HOST_DEC)
            if idx.size == 0:
                continue
            keep = k >= KAPPA_MIN
            if (~keep).any() and idx.size > TOP_PER_SN:
                top = np.argsort(k)[-TOP_PER_SN:]
                keep[top] = True
            for i, ki in zip(idx[keep], k[keep]):
                ls = int(df.ls_id.iloc[i])
                h = hits.get(ls)
                if h is None:
                    hits[ls] = [float(ki), float(ki), 1, prov, int(i)]
                else:
                    h[0] = max(h[0], float(ki))
                    h[1] += float(ki)
                    h[2] += 1


def package_rows(df, hits, prov, rows):
    """Extract runner-compatible photometry for this region's hits."""
    sel = {ls: h for ls, h in hits.items() if h[3] == prov}
    if not sel:
        return
    est_z = df.z_spec.where(df.z_spec.notna(), df.zp_med).to_numpy(float)
    for ls, (kmax, ksum, nsn, _, i) in sel.items():
        g = df.iloc[i]
        row = {"ls_id": ls, "ra": g.ra, "dec": g.dec, "lensed": False,
               "survey_region": prov,
               "z_best": est_z[i], "z_is_spec": bool(pd.notna(g.z_spec)),
               "kappa_max": kmax, "kappa_sum": ksum, "n_sn": nsn}
        for b in ("g", "r", "z", "w1", "w2"):
            row[f"flux_{b}"] = g[f"flux_{b}"]
            row[f"flux_ivar_{b}"] = g[f"flux_ivar_{b}"]
            row[f"mw_transmission_{b}"] = g[f"mw_transmission_{b}"]
        n_nir = 0
        for src, dst in (("j", "j"), ("h", "h"), ("ks", "k")):
            m, e = g.get(f"mag_{src}"), g.get(f"magerr_{src}")
            if pd.notna(m) and pd.notna(e) and e > 0:
                f = 10.0 ** ((22.5 - m) / 2.5)          # nanomaggies, dered
                sig = f * np.log(10.0) / 2.5 * max(e, 0.005)
                row[f"flux_{dst}"] = f
                row[f"flux_ivar_{dst}"] = 1.0 / sig ** 2
                row[f"mw_transmission_{dst}"] = 1.0
                n_nir += 1
            else:
                row[f"flux_{dst}"] = np.nan
                row[f"flux_ivar_{dst}"] = np.nan
                row[f"mw_transmission_{dst}"] = np.nan
        row["n_nir_bands"] = n_nir
        rows.append(row)


def main():
    rng_unused = np.random.default_rng(0)  # parity with driver signatures
    rows = []
    hits_all = {}

    # ---- DES groups X/C/E ------------------------------------------------
    res = pd.read_csv(NIR_KAPPA["des"][0])
    args = SimpleNamespace(n_rand=500, smhm_inverse="posterior",
                           logmh_max=13.8)
    for gname in ("X", "C", "E"):
        gsn = res[res.GROUP == gname]
        fields, center = des_full.FIELD_GROUPS[gname]
        rad = angular_sep_arcsec(center[0], center[1],
                                 gsn.HOST_RA.to_numpy(),
                                 gsn.HOST_DEC.to_numpy()).max() / 3600.0
        cfg = des_full.make_cfg(args, center, rad + 0.25)
        tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
        df = catalog.clean_and_merge(cfg, *catalog.fetch_regional(cfg, tap))
        df = attach_nir(df)
        cl = clu.fetch_clusters(cfg, lambda u: TapClient(u,
                                                         cfg.data.cache_dir))
        hm = HaloModel(cfg.halo_model, cfg.cosmo, 1.15)
        members = clu.assign_members(cfg, df, cl, hm)
        df = df[~members].reset_index(drop=True)
        est = make_estimator("nir_direct_fsf", cfg.cosmo)
        hits = {}
        run_region(cfg, df, gsn, hm, est, hits, f"des:{gname}")
        fresh = {}
        for ls, h in hits.items():
            if ls not in hits_all:
                hits_all[ls] = h
                fresh[ls] = h
            else:
                hits_all[ls][0] = max(hits_all[ls][0], h[0])
                hits_all[ls][1] += h[1]
                hits_all[ls][2] += h[2]
        package_rows(df, fresh, f"des:{gname}", rows)
        log(f"des:{gname}: {len(hits)} contributors "
            f"({len(rows)} packaged so far)")

    # ---- Union3 / Pantheon+ regions --------------------------------------
    for survey in ("union3", "pantheon"):
        res = pd.read_csv(NIR_KAPPA[survey][0])
        reg = pd.read_csv(NIR_KAPPA[survey][1]).set_index("region")
        cosmo = uf.make_cfg((0.0, 0.0), 1.0, 200).cosmo
        hm = HaloModel(uf.HaloModelConfig(), cosmo, uf.Z_SRC_MAX)
        est = make_estimator("nir_direct_fsf", cosmo)
        for rid, grp in res.groupby("region"):
            rrow = reg.loc[rid]
            cfg = uf.make_cfg((rrow.ra, rrow.dec),
                              max(rrow.radius_deg + 0.25, 0.45), 200)
            tap = TapClient(cfg.data.tap_url, cfg.data.cache_dir)
            df = catalog.clean_and_merge(cfg,
                                         *catalog.fetch_regional(cfg, tap))
            df = attach_nir(df)
            cl = (uf.local_clusters((rrow.ra, rrow.dec),
                                    catalog.region_radius_deg(cfg))
                  if uf.WH_LOCAL.exists() else
                  pd.DataFrame(columns=["name", "ra", "dec", "z", "m200"]))
            members = (clu.assign_members(cfg, df, cl, hm) if len(cl)
                       else np.zeros(len(df), dtype=bool))
            df = df[~members].reset_index(drop=True)
            hits = {}
            run_region(cfg, df, grp, hm, est, hits, f"{survey}:{rid}")
            fresh = {}
            for ls, h in hits.items():
                if ls not in hits_all:
                    hits_all[ls] = h
                    fresh[ls] = h
                else:
                    hits_all[ls][0] = max(hits_all[ls][0], h[0])
                    hits_all[ls][1] += h[1]
                    hits_all[ls][2] += h[2]
            package_rows(df, fresh, f"{survey}:{rid}", rows)
            log(f"{survey}:{rid}: {len(hits)} contributors "
                f"({len(rows)} packaged)")

    out = pd.DataFrame(rows)
    out = out.sort_values("kappa_max", ascending=False).reset_index(
        drop=True)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    n_fb = int((out.n_nir_bands >= 2).sum())
    log(f"saved {OUT}: {len(out)} unique contributors, "
        f"{n_fb} with >=2 NIR bands (FB-ready), "
        f"median kappa_max {out.kappa_max.median():.4f}")


if __name__ == "__main__":
    main()
