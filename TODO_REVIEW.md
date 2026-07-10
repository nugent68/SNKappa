# SNKappa / DES lensing paper — review findings & work plan

## IMPLEMENTATION STATUS (2026-07-10)

NOTE: the manuscript now lives ONLY in the Overleaf project (git remote);
`scripts/make_figs.py` (moved from paper/) regenerates its figures into
`output/figs/` and its statistics into `output/des_full/paper_stats.json`.
References to `paper/...` below are historical.

All P0–P4 items below are implemented and the pipeline + paper are updated.
Post-fix headline (P(Ia)>0.9, N=1450): slope = −2.00 ± 0.49,
A = 0.92 ± 0.23 (stat) ± 0.09 (sys); permutation null p < 1e-4 (3.8σ).

- P0: 0.1 fixed (shared `snkappa/fitting.py` + `tests/test_fit_weighting.py`);
  0.2 resolved as option (b) — CENTERED mass scatter, and the per-sightline
  cluster MC replaced by deterministic precomputed miscentering-convolved
  profiles (`snkappa/batch.py:ClusterField`); 0.3 fixed; 0.4 ported
  (randoms count guard + per-SN `area_frac`/`area_flag` columns).
- P1: 1.1 permutation null + spatial block bootstrap in `paper/make_figs.py`
  (permutation σ=0.55 > iid bootstrap 0.49, as predicted — headline
  significance now quoted from the permutation test); 1.2 all three tests
  clean (excision variant −2.05±0.52; κ vs HOST_LOGMASS ρ=0.008 p=0.77;
  mass-step covariate moves slope <0.3%); 1.3 X+S spec-on −0.66±1.01 vs
  photo-only −0.51±1.00 → no spectroscopy needed; 1.4
  `scripts/photoz_validation.py` (cov68=0.65–0.66, outliers 1%, 5% in
  faintest half-mag; outlier-floor hook added to `photoz.grid_pdf`, not
  activated); 1.5 W1 fraction 85–87%, W1-only slope −1.92±0.52.
- P2: 2.1 `scripts/attenuation_mc.py` → λ_gal=0.97, λ incl. clusters
  0.7 (mean) / 0.99 (median) → A_true ≈ 0.9–1.3 (in paper); 2.2 posterior
  <Mh|M*> with Despali16 HMF prior is now the default (`smhm_inverse`);
  2.3 systematics table in the paper (quad. total 0.09); 2.4 discussed as
  caveat (cap-insensitivity Δ<0.01 bounds it; mock calibration flagged).
- P3: all done (3.1 Om0=0.352 everywhere + per-zbin demean; 3.2 exact
  (1−κ)^−2 Δμ in Table 1, shear omission stated; 3.3 σ_mis = 0.2 r500;
  3.4 z_src interpolation; 3.5 500 randoms; 3.6 de-lensed-weights
  robustness row; 3.7 z<1 row; 3.8 fixed; 3.9 `snkappa/batch.py`).
- P4: all text items done; Data Availability now true (.gitignore
  exception for the catalog + `des_pilot/README.md` provenance).
  NOTE: new/changed files are NOT yet committed — review `git status`.
- P5: 5.1 DONE on Perlmutter (repo m2218) with cosmoDC2 v1.1.4:
  `scripts/mock_extract_dc2.py` + `scripts/mock_calibration.py` +
  `scripts/mock_dc2.sbatch` (debug queue, 3 variants in parallel, ~5 min).
  Results (1950 sightlines, 6 healpix regions, output/mock_dc2/):
  noiseless A=0.45±0.04, fiducial (DES-like noise) A=0.31±0.03, naive
  SMHM A=0.22±0.02 → λ_mock = 0.68±0.09 (independently confirms the
  local attenuation MC's λ≈0.7) and the debiased SMHM inversion is
  worth +39% in recovered amplitude. CAVEAT: absolute A_mock is not a
  model test — cosmoDC2 convergence lives on 0.5' HEALPix shells and is
  suppressed at the <~1 Mpc one-halo scales carrying the signal
  (Korytov+2019; degraded at R<~1 Mpc per Kovacs+2022). RESOLVED by the
  Phase-0 resolution-matched closure (docs/MICE_PLAN.md): smoothing the
  PREDICTION with a Gaussian kernel recovers A_matched = 0.57/0.71/0.84/
  0.93 (±0.04–0.08) at sigma = 0.25'/0.5'/0.85'/1.3' — monotonically ->
  1 at the plausible effective map resolution, i.e. the DC2 deficit is a
  resolution artifact and the model amplitude closes at map scales
  (output/mock_dc2/mock_summary_sm*.json).
- P5.2-analog (absolute one-halo closure vs DATA): in progress via
  scripts/delta_sigma_closure.py — predicted DeltaSigma for the DESI DR1
  "Lensing Without Borders" lens bins (BGS/LRG; measurements + lens
  catalogs with LS photometry on NERSC at desicollab/science/c3/
  DESI-Lensing, unblinded v1.5) vs DES/KiDS/HSC/SDSS sources. NOTE:
  collaboration data — check DESI publication policy before quoting in
  the paper (the public release repo is still a stub). 5.3–5.5 open.

---

Source: code + paper review of github.com/nugent68/SNKappa (2026-07-10).
Audience: a Claude session running in the repo where the code, the DES input
files (`des_pilot/DES_HD.csv`, `des_pilot/DES_meta.csv`), the cached catalogs,
and the paper (`paper/des_lensing_apjl.tex`) are set up.

Context: `scripts/des_full.py` predicts per-SN external convergence kappa_ext
for DES-SN5YR and regresses published Hubble residuals on it;
`paper/make_figs.py` produces the ApJL numbers/figures. Headline numbers in
the current draft: slope = -2.10 +/- 0.47, A = 0.96 +/- 0.22 (4.4 sigma).
Tasks below are ordered by priority within each section. Tasks marked
**[RERUN]** change published numbers — after completing all of them, rerun
`scripts/des_full.py` and `paper/make_figs.py` once and update every number in
the paper (abstract, Sec. 4, figures, Table 1) in a single pass.

---

## P0 — Bugs that affect published numbers

### 0.1 Fix the regression weighting **[RERUN]**
`np.polyfit(x, y, 1, w=...)` expects `w = 1/sigma` (weight multiplies the
UNSQUARED residual — see numpy docs). The code passes `w = 1/MUERR**2`,
i.e. an effective sigma^-4 weighting that over-weights low-z SNe where the
lensing signal is smallest.
- Fix in `scripts/des_full.py` `fit()` (~line 270: `ww = 1/d.MUERR**2` → `1/d.MUERR`)
- Fix in `paper/make_figs.py` `wslope()` (~line 36), including inside the bootstrap loop.
- NOTE: the `np.average(..., weights=1/sigma**2)` calls (mean subtraction,
  binned means, hi/lo split) are CORRECT — do not change those.
- Also fix the same pattern in `scripts/des_pilot.py` if present.
- Add a regression test (e.g. `tests/test_fit_weighting.py`): synthetic data
  with heteroscedastic noise; the weighted slope must match an analytic
  weighted-least-squares solution.

### 0.2 Decide + document the cluster mass-scatter Jensen boost **[RERUN]**
`cluster_kappa_marg()` in `scripts/des_full.py` (~line 183) averages kappa over
lognormal mass draws (0.25 dex): E[10^(0.25 N)] = 1.18, silently boosting every
cluster's effective mass ~18%. Either (a) keep it as a deliberate Eddington-type
correction and document it in code + paper Sec. 3, or (b) center the draws so
E[M] = M_catalog (`10**(rng.normal(...) - 0.5*(CL_MSCAT*ln10)**2 ...)`). Pick one
explicitly. Also raise `CL_ND` from 64 (per-sightline MC jitter attenuates the
regression) — either >= 512 draws, or better: precompute a miscentering-convolved
Sigma(R) profile once per (mass, z) so the cluster term is deterministic.

### 0.3 Concentration computed from uncapped halo mass
`snkappa/halos.py` `halo_params()` (~lines 210–217): mass is capped at
`logmh_max=13.8` but `concentration.concentration()` is called with the
UNCAPPED `lmh`. Cap first, then compute c. Small effect; verify the change on
one field before the full rerun.

### 0.4 Port the masked-randoms / area guards into des_full **[RERUN — catalog values]**
`snkappa/randoms.py` flags randoms with aperture counts < 0.5 * median and
excludes them from the zero point; `scripts/des_full.py` (~lines 243–245) has
no such guard, and SN sightlines get no `catalog.area_fraction()` check.
This biases the RELEASED per-SN kappa_ext values and Fig. 2 (slope is immune
to a per-bin zero-point shift, but the catalog is the product).
- Apply the ngal guard to the 120 randoms per group.
- Compute `area_fraction` per SN sightline; add a flag column to
  `des_all_kappa.csv`; state the flag in the paper's catalog description.

---

## P1 — Referee-critical robustness tests (cheap, decisive)

### 1.1 Permutation-null significance
Shuffle Hubble residuals among SNe WITHIN z-bins (preserves both marginals and
the kappa spatial structure), refit the slope ~10^4 times, quote the empirical
p-value. This absorbs the spatial covariance between overlapping 10-arcmin
apertures (shared foreground clusters) that the current i.i.d. bootstrap
ignores — the quoted 4.4 sigma is an upper bound until this exists. Also add a
spatial block bootstrap (blocks >= aperture scale) as a cross-check. Report in
the paper alongside the bootstrap error.

### 1.2 Host-environment confounder test
Galaxies physically associated with the SN host enter kappa_ext through the
low-z tail of their photo-z PDFs, and SN standardized brightness correlates
with host environment (mass step) — same sign structure as the lensing signal.
Three tests, all cheap:
- (a) Rerun with foreground galaxies excised when photo-z is consistent with
  the SN: |z - z_SN| < 0.1*(1+z_SN) (efficiency-weighted signal loss is tiny
  since the lensing kernel vanishes there). **[RERUN variant — report as
  robustness row, not headline]**
- (b) Regress kappa_ext against host stellar mass (host masses are in the
  DES-SN5YR release); should be consistent with zero.
- (c) Refit the slope with a host-mass-step covariate; slope should be stable.
Add the outcome to the paper (new robustness paragraph or table).

### 1.3 Spec-z on/off test in X+S (answers the "drop non-DESI fields?" question)
Do NOT drop CDF-S/Elais-S1 (Fig. 3 shows C+E carry the signal; dropping them
costs sqrt(2) statistics for no gain). Instead rerun the X and S groups with
DESI spec-z deliberately ignored (photo-z only) and compare slopes. Unchanged →
demonstrates the method needs no spectroscopy (LSST scaling argument);
changed → the spec fields become the calibration anchor. Either result goes in
the paper.

### 1.4 Photo-z p(z) empirical validation using the DESI overlap
Use the ~33k DESI DR1 spec-z in X/S as truth: PIT / coverage test of the
reconstructed split-normal p(z) (`snkappa/photoz.py`) at the z<=22.5 faint end;
measure the catastrophic-outlier rate; if needed add a small uniform outlier
floor component to `grid_pdf()` and `sample()`. Report the coverage numbers in
the paper (one sentence + appendix-level detail).

### 1.5 Quantify the Taylor2011 fallback population
Galaxies failing the W1 SNR>2 cut (`snkappa/catalog.py` ~line 115) fall back to
the optical-color estimator whose flat-SED K-correction is known-biased at
z >~ 0.7 (per the docstring in `snkappa/stellar.py`). Measure the W1-detected
fraction vs mag and z; rerun the slope on a W1-required subsample; report the
amplitude stability.

---

## P2 — Interpretation of A (the paper's central claim)

### 2.1 Errors-in-variables attenuation
Noise in predicted kappa (photo-z, 0.16 dex M*, SMHM scatter, cluster MC
jitter, zero-point noise ~ sigma_kappa/sqrt(120) per bin) attenuates the fitted
slope: A_fit = A_true * lambda, lambda = Var(signal)/Var(total). A = 0.96 could
be (normalization high) x (lambda < 1). The single-SN MC machinery
(`snkappa/montecarlo.mc_kappa_raw`) computes exactly the needed per-sightline
noise variance but `des_full.py` never uses it (fiducial kappa only).
- Minimum: estimate per-SN Var_noise(kappa) from the MC for a representative
  subsample, compute lambda, report both A_fit and A_fit/lambda with the
  attenuation uncertainty.
- Better: latent-variable (Bayesian errors-in-variables) regression using
  per-SN P(kappa). This is the single most valuable upgrade to the paper.

### 2.2 SMHM inversion is not <Mh|M*> (Eddington bias)
`snkappa/halos.py` inverts the MEAN Behroozi relation and adds symmetric
scatter in log Mh. Above the knee (dlogM*/dlogMh ~ 0.2–0.3) the true <Mh|M*>
lies below the naive inversion by ~0.1–0.2 dex (low-mass halos scattering up
outnumber). With kappa ~ M^0.7–0.8 that is a 15–40% amplitude systematic —
same size as the statistical error. Implement P(Mh|M*) ∝ P(M*|Mh) n(Mh) with a
colossus halo mass function prior (per z-bin lookup table; drop-in replacement
for `_build_smhm_inverse`). Note the partial cancellation with the ignored
forward Jensen scatter (~+10%) and make it deliberate.

### 2.3 Systematic error budget table for A
New paper table: SMHM inversion (2.2), M*/L calibration (0.16 dex → dA),
logmh_max cap, cluster mass scale incl. 0.2 boost decision, miscentering,
photo-z outliers (1.4), attenuation (2.1). Until this exists the "bounds the
halo-mass normalization at ~30%" claim in Sec. 5 is not earned.

### 2.4 Mass gap between the cap and the cluster catalog
Single galaxies capped at 10^13.8; Wen & Han complete only above ~M500 of
0.5–1e14. Group halos in 10^13.8–10^14.2 are systematically under-massed.
Quantify (e.g. HMF-weighted missing-kappa estimate), and/or inject a DESI
photometric group catalog (Zou et al. / Yang et al. DESI groups) as a variant
run.

---

## P3 — Smaller model/consistency fixes

- 3.1 Cosmology alignment **[RERUN]**: kappa predicted with Om0=0.3, H0=70
  (config defaults in `make_cfg`) while residuals use Om0=0.352
  (`des_full.py` ~line 259). Align both to the DES-SN5YR values. Also demean
  HR per z-bin (kills residual cosmology/calibration z-trends; free, since
  kappa_ext is already per-bin zero-pointed).
- 3.2 Exact tail magnification: use Delta_mu = -2.5 log10[1/((1-k)^2 - gamma^2)]
  (or at least the (1-k)^-2 form) for Table 1 predictions; the linear
  -2.171*kappa is 5–10% off at kappa ~ 0.1. Compute shear gamma from the same
  halo sum if feasible, else state the omission.
- 3.3 Miscentering: scale the 30 arcsec Rayleigh with cluster z (fixed physical
  scale, e.g. ~0.15–0.4 r500) instead of fixed angle.
- 3.4 Use actual z_SN instead of bin center: linear interpolation of kappa
  between the two adjacent z_src bin evaluations (cheap; removes a +/-0.025
  z_src quantization).
- 3.5 More randoms: 120 → >= 500 per group (total runtime is ~26 min; this is
  cheap and shrinks per-bin zero-point noise).
- 3.6 MUERR already contains DES's sigma_lens = 0.055z term → weights
  down-weight the signal carriers; add a footnote, optionally show a variant
  with the lensing term removed from the weights in quadrature.
- 3.7 Malmquist/magnification selection: add a z < 1.0 robustness row.
- 3.8 `photoz.sample`: the `(z > 0).all()` retry check essentially never
  passes for large arrays (dead code before the clip). Harmless; tidy or
  remove.
- 3.9 Refactor: pull `BatchEngine` + the des_full loop into the `snkappa`
  package so the batch path shares the single-SN pipeline's guards (area
  checks, MC, provenance) instead of drifting as a parallel implementation.

---

## P4 — Paper text corrections (do after reruns so numbers are final)

- 4.1 **Data Availability is currently false**: points to
  `output/des_full/des_all_kappa.csv` but `output/` is gitignored and the DES
  input prep isn't in the repo. Commit the kappa catalog (small) + an input
  prep script, or mint a Zenodo DOI and cite it.
- 4.2 Sec. 3: "120 random sight lines per field per source-redshift bin" →
  actually 120 positions per FIELD GROUP, drawn once, reused across all bins
  (a feature — coherent zero points — but describe it accurately).
- 4.3 Fig. 2 discussion: sigma_kappa(randoms) includes photo-z/halo-model
  noise, so "catalog halos account for 40–50% of expected dispersion" is an
  upper bound; the honest de-lensable variance is the correlated part
  (~ A * sigma_pred^2). Reword.
- 4.4 Method section must state: the regression estimator + weights (post-fix),
  3 arcsec inner exclusion, 10 arcmin aperture, 30 arcmin cluster radius,
  z<=22.5 catalog limit, bin-center z_src approximation (or 3.4's fix),
  the cluster mass-boost decision (0.2).
- 4.5 "Consistent with unity at the 4% level" → ambiguous (reads as p-value);
  say "central value within 4% of unity, well within the 22% uncertainty."
- 4.6 Spearman p = 1e-4 is ~3.9 sigma — supporting, not "confirming," 4.4
  sigma; once 1.1's permutation test exists, quote that as the independent
  significance.
- 4.7 Add the robustness results (1.1–1.5) and the systematics table (2.3).

---

## P5 — Larger extensions (post-submission or referee-response ammunition)

- 5.1 End-to-end mock calibration: run the identical pipeline on a simulated
  lightcone (MICE / Buzzard / CosmoDC2) with known true kappa along DES-like
  sightlines. Calibrates attenuation, SMHM-inversion bias, the cap, and the
  missing 2-halo term in one exercise — converts every "should be small" into
  a number.
- 5.2 Cross-validate against DES Y3 shear-based mass maps: correlate predicted
  kappa with map kappa at the SN positions (independent, source-free check of
  the prediction itself).
- 5.3 Add a 2-halo term (linear bias x matter correlation) to the halo model.
- 5.4 SZ cross-check of Wen & Han masses where SPT/ACT overlap (SPT deep field
  covers CDF-S).
- 5.5 Deeper photometry in the SN fields (VIDEO NIR, HSC-Deep in XMM-LSS) for
  masses/photo-z at the faint end; LSST-ready path.

---

## Verified-correct during review (do not "fix")

- Behroozi+13 and Moster+13 SMHM coefficient implementations match the
  published parametrizations.
- Sigma_crit convention (validated by the SIS Einstein-radius unit test),
  BMO truncated-NFW profile + interpolation table, cylinder-mass amplitude
  test.
- Nir1um rest-1um K-correction sign/form (Hogg 2002 band-shift) and the
  z<->W1 log-interpolation logic.
- Data Lab -9999 sentinel handling in `dered_mag`.
- `np.average(..., weights=1/sigma**2)` uses (weighted means) — correct; only
  the `np.polyfit` weight convention is wrong (0.1).
- Spec-z >= z_src galaxies removed rather than falling back to photo-z
  (`catalog.clean_and_merge`) — correct choice.
