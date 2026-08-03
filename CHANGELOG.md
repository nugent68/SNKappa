# Changelog

Public record of how the analysis evolved, connecting every number in the
manuscript to the commits and tracked artifacts that produced it. Code
comments of the form `(TODO n.m)` refer to the review-checklist item
numbers listed at the bottom of this file. NOTE: on 2026-08-02 the
repository history was rewritten to excise DESI-internal measured
DeltaSigma vectors inadvertently committed on 2026-07-10; commit hashes
from that date forward changed (references below are the current ones).

## 2026-08-02 — Validation (v) re-anchored on a fully public closure

Measured ΔΣ for BOSS DR12 LOWZ/CMASS lenses with the public KiDS-Legacy
DR5 shear catalog (`boss_kids_measure.py`, dsigma recipe, 100-field
jackknife; little-h conventions converted explicitly) and closed the
halo chain over the identical lenses (`boss_closure.py`):
A_1h = 0.94 ± 0.04 / 1.07 ± 0.06 (LOWZ z 0.15–0.43) and 1.18 ± 0.05 /
1.16 ± 0.06 (CMASS z 0.43–0.70) for the fiducial chain; legacy constant
M*/L gives 1.9–2.1 (the mass-scale diagnosis now stands on public data
alone); naive SMHM inversion 0.55–0.62 (excluded). Matches the
DESI-internal LWB pattern; the manuscript's validation (v) now cites
only public, end-to-end reproducible inputs and the DESI
publication-policy dependency is removed.

## 2026-08-02 — History scrub: DESI-internal ΔΣ vectors removed

The closure JSONs had embedded the measured ds_meas/ds_err vectors of
the DESI Lensing Without Borders v1.5 internal data products alongside
our predictions — collaboration-internal data pending the DESI release.
Tracked files replaced by prediction-only `closure_pred*.json`; the
three original files removed from ALL history (git-filter-repo); cited
hashes remapped. The derived closure ratios quoted in the manuscript
remain subject to DESI publication-policy sign-off.

## 2026-08-02 — Pantheon+ capstone: GLS covariance, standardization test, DESI trio

Pantheon+ (Scolnic et al. 2022) run through the generalized survey
driver (`union3_full.py --hd/--outdir`; per-SN distances released, no
Tripp needed; 907 sightlines, one genuinely-uncovered low-b region
fails cleanly). Three capstone results:

- **Full-covariance GLS slope** (new `fitting.gls_slope`, released
  1701² stat+syst matrix): **−2.62 ± 0.79 (3.3σ)**, A = 1.21 ± 0.37 —
  confirms the diagonal fit (−2.40 ± 0.62); errors grow only ~27%.
  The "diagonal errors" caveat carried by all previous slopes retires:
  calibration covariances neither create nor destroy the signal.
- **Standardization independence at fixed κ**: 530 sightlines matched
  to Union3 (κ correlation 0.993); slopes agree (−2.48 ± 0.66 vs
  −2.76 ± 0.77) and the residual-difference regression is
  **+0.08 ± 0.31** — the lensing signal is independent of the SN
  standardization pipeline (SALT2/BBC vs Tripp-on-UNITY).
- **DESI DR2 trio complete**: de-lensing shifts the evolving-DE
  ΔΔχ² by +0.38 (DES leg, 4.19→4.24σ), +0.19 (Union3 leg,
  3.76→3.79σ), −0.10 (Pantheon+ leg, 2.82→2.81σ). Lensing selection
  does not materially modify the preference for ANY of the three SN
  compilations.

Standalone Pantheon+: slope −2.40 ± 0.63 (3.8σ), exact-prediction
A = 1.08 ± 0.29, A_gal = 1.05 ± 0.49 / A_cl = 1.14 ± 0.36,
permutation z = 2.86. Artifacts under `output/pantheon/` and
`output/union3/desi_leg.json`; data in gitignored `data_pantheon/`.

## 2026-08-02 — Union3 predicted-κ measurement (first pass)

Full survey-mode run on the Union3 compilation (Rubin et al. 2025, ApJ
986, 231 — the 3.8σ SN leg of DESI DR2), from the released UNITY inputs:
Tripp-standardized residuals (`union3_prep.py`; α = 0.122, β = 2.57,
per-sample σ_int), FoF survey regions (`snkappa/regions.py`; 212
processed, 3 masked-edge singletons failed cleanly), cluster tier from
the full local Wen & Han CDS table (VizieR-independent), z_src to 1.60.
Results (1,213 field SNe; See-Change cluster-targeted sample excluded
from fits by design):

- slope dΔμ/dκ = **−2.10 ± 0.63 (3.4σ)** — beats the 2.8σ audit
  forecast; exact-prediction fit A = 0.94 ± 0.29; two-component
  A_gal = 0.67 ± 0.40, A_cl = 1.14 ± 0.38; permutation z = 2.45.
  First predicted-κ lensing test of the PS1 (−3.83 ± 1.07, 3.6σ
  standalone), SDSS, SNLS, and ESSENCE sightlines.
- **Joint DES + Union3: A = 0.78 ± 0.17 (4.6σ)** (DES3 overlap removed
  from the Union3 side).
- **Cross-survey validation: κ correlation 0.994** on 109 DES3
  sightlines predicted independently by both pipelines (different
  catalogs fetches, z-grids, randoms).
- See-Change (cluster-targeted): ⟨κ_ext⟩ = +0.0015 vs field +0.0004 —
  elevated as selection predicts (modest because their targeting
  clusters sit near the SN plane where lensing efficiency vanishes).
Artifacts: `output/union3/` (hd, kappa catalog, regions, fit summary,
Tripp params). Data: `data_union3/` (gitignored; MANIFEST.txt).

## 2026-08-01 — Measurement-improvement sprint (shear, robustness, moments, de-lensing)

- **Shear + exact magnification**: tangential shear vector-summed
  (spin-2) over both tiers from the analytic BMO ΔΣ; ClusterField shear
  from the miscentering-convolved profile's own ΔΣ. New catalog columns
  `gamma`, `dmu_pred` (exact 2.5 log10[(1−κ)²−|γ|²], random-zero-pointed).
  ⟨|γ|⟩ = 0.0083 adds ~4% to the prediction dispersion; exact-prediction
  fit **A = 0.757 ± 0.192** (vs κ-only 0.786 ± 0.195); κ columns
  bit-identical; zero strong-lensing clips. Table 1 Δμ_pred now includes
  shear.
- **Latent-fit robustness** (`latent_fit.py --robustness`): jackknife
  [1.12–1.66], z<1 and de-lensed-weights rows stable; EIV permutation
  null z = 2.31 — the latent fit is the amplitude estimator, not the
  detection statistic (detection remains the WLS permutation, 3.7σ).
- **Moments channels** (`moments_test.py`): honest nulls with diagnosis —
  the variance channel is degenerate with the MUERR calibration (freeing
  it gives c0 = 0.86, i.e. DES errors over-cover ~14% in this subsample,
  and a = 0 ± 1.2 for the z-growing part); skew channels are noise-
  limited (predicted 0.01–0.03 vs ~0.2 noise per bin). LSST-scale
  samples needed for the moments channels.
- **Variant harmonization**: the 2026-07-11 variant sweep had used
  n_rand = 200 (vs 500 for the headline); all six variants rerun at 500
  with the new columns. Table-2-class slopes shift at the second decimal
  (e.g. excise −1.93 → −1.91, naive −1.25 → −1.23); the nospecz variant
  is X+S-restricted as published.
- **De-lensing demo** (`delens_demo.py`): ΔΩ_M = −0.002 (ΛCDM,
  negligible); diagonal SN-only wCDM shifts Δw = +0.09 — driven by the
  mean SN-vs-random overdensity (⟨κ_ext⟩_w = +0.0019 at z > 0.7, a
  +3.3 mmag coherent high-z correction); de-lensing raises the z > 0.7
  skewness by +0.17 (removal of the magnified bright tail). Differential,
  no-covariance statement only.

## 2026-08-01 — Latent-variable (errors-in-variables) regression

`snkappa/latent.py` + `scripts/latent_fit.py`: empirical-Bayes EIV fit of
the amplitude with per-SN Monte-Carlo noise variances for all 1450 sight
lines (`output/des_full/latent_noise.csv`), replacing the external
lambda correction. The prediction noise is split into a CLASSICAL part
(M* measurement error, richness-mass error, zero-point sampling — drives
the shrinkage) and a BERKSON part (photo-z p(z), SMHM/concentration
intrinsic scatter, miscentering — already marginalized by the prediction;
enters the residual only and does not attenuate). Misclassifying Berkson
variance as classical inflates A (1.78 vs 1.38 here) — unit-tested.
Result: **A = 1.38 ± 0.54** (single), A_gal = 1.55 ± 0.67 /
A_cl = 1.68 ± 0.85 (two-component) — consistent with unity and with the
mock-based de-attenuation of the naive fit (0.79/0.68 ≈ 1.2). Estimator
validated on synthetic latents at the observed kappa skewness (bias
+2 ± 2% at skew 4.8). Artifacts: `output/des_full/latent_fit.json`.

Also: two-component WLS amplitudes (2026-08-01, `0f7fac5`):
A_gal = 0.85 ± 0.32, A_cl = 0.76 ± 0.27, components uncorrelated
(r = 0.015); catalog gains kappa_gal_ext / kappa_cl_ext columns.

## 2026-07-11 — Post-review reruns; all gates passed (`01da651`)

All published numbers regenerated after the `b0f3998` fixes:

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

## 2026-07-10 — Cluster-tier excision + catalog fixes (`b0f3998`)

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

## 2026-07-10 — M*/L recalibration to the DESI mass scale (`da1ce3c`–`9eda5a9`)

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
