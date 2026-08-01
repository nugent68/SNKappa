# Changelog

Public record of how the analysis evolved, connecting every number in the
manuscript to the commits and tracked artifacts that produced it. Code
comments of the form `(TODO n.m)` refer to the review-checklist item
numbers listed at the bottom of this file.

## 2026-07-11 — Post-review reruns; all gates passed (`dc74d8b`)

All published numbers regenerated after the `c652439` fixes:

- Headline unchanged to quoted precision: slope dΔμ/dκ_ext = −1.71 ± 0.42,
  A = 0.786 ± 0.196 (the catalog-cut fix added only ~0.2% of galaxies per
  region).
- Cluster-inclusive host-plane excision: −1.93 ± 0.47 (no collapse — the
  cluster-plane confounder is excluded).
- New `no_cluster_at_sn_plane` row: −1.16 ± 0.50 (N=1273); the 177 dropped
  cluster-proximate SNe carry slope −3.82 ± 0.76, persisting under fully
  excised κ — i.e. real signal concentrated near clusters, not confounding.
- Permutation null rerun at 10^5 shuffles: p = 6×10⁻⁵ (3.7σ), no longer
  floored by the shuffle count.
- `output/des_full/attenuation.json` now tracked: λ_gal = 0.89; the paper
  adopts the mock-calibrated λ = 0.68 ± 0.09 → A_true ≈ 0.8–1.2.

## 2026-07-10 — Cluster-tier excision + catalog fixes (`c652439`)

- The `--excise-host` robustness variant now also drops clusters with
  |z_cl − z_src| < 0.1(1+z_src): cluster-catalog redshift errors could
  otherwise let the SN's own host cluster pass the foreground cut, and
  clusters dominate the κ_ext tail.
- Per-SN `cl_dz_min_2am` column (nearest-cluster plane proximity within
  2′) + the `no_cluster_at_sn_plane` robustness row.
- Catalog requires z + (W1 or g) instead of g unconditionally (the old cut
  dropped g-undetected massive red galaxies the rest-1μm estimator handles
  via z+W1).
- DESI spec-z crossmatch resolves fiber collisions by distance.

## 2026-07-10 — M*/L recalibration to the DESI mass scale (`da1ce3c`–`ce7fe02`)

- Galaxy–galaxy lensing closure test (`scripts/delta_sigma_closure.py`):
  predicted ΔΣ for DESI DR1 BGS/LRG lens bins vs DES/KiDS/HSC/SDSS
  sources showed the constant M*/L_1μm = 0.6 under-predicts by 1.6–2.0×
  at the massive end.
- Driver pinned by per-galaxy comparison with the public FastSpecFit VAC
  (500k DESI galaxies): rest-1μm masses run 0.2–0.25 dex low for massive
  red galaxies. New default estimator `nir1um_fsf` applies a sigmoid
  Δ(logM*, z) correction; recalibrated closure gives A_1h ≈ 0.96–1.08
  (BGS), 1.12–1.21 (LRG 0.4–0.6).
- Recalibrated DES headline: slope −1.71 ± 0.42, A = 0.79 ± 0.20 (stat)
  ± 0.12 (sys). Note the recalibration moved A *away* from unity
  (0.92 → 0.79): the mass scale was frozen on external lensing data, not
  tuned to the Hubble residuals.

## 2026-07-10 — cosmoDC2 end-to-end mock calibration (`e2586f1`–`ba9350c`)

Identical pipeline run on cosmoDC2 truth (1950 sightlines, 6 regions) with
DES-like forward-modeled noise: attenuation λ_mock = 0.68 ± 0.09;
debiased SMHM inversion worth +39% in recovered amplitude; smoothing the
prediction to the map resolution recovers A_mock = 0.93 ± 0.08 at a 1.3′
kernel (the DC2 absolute deficit is a HEALPix-shell resolution artifact).
Artifacts in `output/mock_dc2/`.

## 2026-07-10 — Review-driven overhaul (`17fa494`)

Full internal code + paper review; the fixes that changed published
numbers:

- Regression weighting: `np.polyfit` weight convention corrected
  (w = 1/σ, not 1/σ²) — shared `snkappa/fitting.py` + regression test.
- Cluster tier: centered lognormal mass scatter (E[M] = M_catalog; the
  uncentered draw silently boosted every cluster mass 18%), deterministic
  precomputed miscentering-convolved profiles (physical scale 0.2 r500),
  replacing per-sightline MC jitter.
- SMHM inversion: Eddington-debiased posterior ⟨M_h|M*⟩ with a
  Despali+16 halo-mass-function prior (naive inversion kept as a variant;
  it over-predicts the measured galaxy–galaxy lensing and is excluded).
- Concentration computed from the capped halo mass.
- Statistics: permutation null within z bins (preserves the κ spatial
  structure), 0.5° spatial block bootstrap, host-mass confounder tests,
  host-plane excision variant, spec-z on/off, W1-only, photo-z p(z)
  validation against DESI (coverage 65–66%, outliers 1%).
- Consistency: one cosmology (Ω_M = 0.352) for prediction and residuals,
  per-z-bin demeaning, κ interpolated to each SN's z_HD, 500 randoms per
  field with low-count guard, per-SN unmasked-area flags.
- Headline moved from −2.10 ± 0.47 (A = 0.96 ± 0.22) to −2.00 ± 0.49
  (A = 0.92 ± 0.23).

## 2026-07-08/09 — DES-SN5YR survey analysis + manuscript (`2b3a5e0`–`ed22e7e`)

X-field pilot, then all ten DES fields: predicted per-SN κ_ext vs
published DES-SN5YR Hubble residuals; first ApJL draft.

## 2026-07-02/03 — Single-sightline pipeline (`d464910`–`019338c`)

κ_ext estimator for strongly lensed SNe (SN 2025wny environment analysis;
SN 1997ff test case): Data Lab catalog chain, rest-1μm stellar masses,
truncated-NFW halos, random-sightline zero point, Monte Carlo P(κ_ext),
FrankenBlast hybrid-mass machinery (fitted, not adopted — see README),
cluster-halo tier.

---

## Review-checklist item numbers (referenced by code comments)

P0 (bugs affecting published numbers): 0.1 polyfit weight convention ·
0.2 cluster mass-scatter centering · 0.3 concentration from capped mass ·
0.4 randoms/area guards. P1 (robustness): 1.1 permutation null + block
bootstrap · 1.2 host-environment confounder tests (a: excision, b: κ vs
host mass, c: mass-step covariate) · 1.3 spec-z on/off · 1.4 photo-z p(z)
validation · 1.5 W1-fallback population. P2 (interpretation): 2.1
errors-in-variables attenuation · 2.2 Eddington-debiased SMHM inversion ·
2.3 systematic budget · 2.4 group-mass gap. P3 (consistency): 3.1
cosmology alignment + per-bin demeaning · 3.2 exact tail magnification ·
3.3 physical miscentering scale · 3.4 z_src interpolation · 3.5 500
randoms · 3.6 de-lensed weights · 3.7 z<1 row · 3.8 photo-z sampler tidy
· 3.9 batch-engine refactor. P5 (extensions): 5.1 cosmoDC2 mock
calibration · 5.2 mass-map cross-check (open) · 5.3 2-halo term (open) ·
5.4 SZ cross-check (open) · 5.5 deeper photometry (open).
