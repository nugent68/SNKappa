# Mock-lightcone follow-up plan (post-cosmoDC2)

Status: PLAN (2026-07-10). Follows the cosmoDC2 calibration
(`output/mock_dc2/`, TODO_REVIEW P5.1), which delivered
λ_mock = 0.68 ± 0.09 and the +39% SMHM-debiasing validation but could
not test the ABSOLUTE model amplitude: cosmoDC2's ray-traced
convergence lives on Nside=4096 HEALPix shells (0.5′ pixels) and is
suppressed at the ≲1 Mpc one-halo scales that carry our signal
(Korytov et al. 2019; degraded at R ≲ 1 Mpc per Kovacs et al. 2022).

## The resolution constraint that shapes this plan

MICE is NOT the resolution fix. MICE-GC lensing is built with the
"onion universe" method on Nside=4096 all-sky maps — **0.85′ pixels**
(Fosalba et al. 2015, MNRAS 447, 1319), i.e. *coarser* than cosmoDC2.
Any per-galaxy `kappa` served by MICECAT inherits that smoothing. So the
plan splits the goal in two:

1. **Resolution-matched amplitude test** (works on smoothed truth):
   smooth OUR prediction with the simulation's pixel window and demand
   A_matched ≈ 1. Tests everything except the sub-pixel core signal.
2. **Absolute one-halo closure** (resolution-free): calibrate against
   *measured* galaxy–galaxy lensing rather than a simulation.

MICE remains valuable for what it uniquely adds: an independent
galaxy–halo connection (SHAM vs cosmoDC2's Galacticus), native
evolution-corrected DES griz photometry (realistic selection and a
color-based estimator test), a 5000 deg² octant (thousands of
independent DES-like regions → negligible statistical error on A_mock,
λ, and the 2-halo deficit), and halo/cluster info for the cluster tier.

## Phase 0 — resolution-matched closure on cosmoDC2 (no new data; ~1 session)

- Add a `--smooth-arcmin` option to the batch prediction: convolve each
  halo's Σ(R) with a Gaussian/pixel window of 0.5′ (DC2) at the lens
  D_A before evaluating κ_pred. (Implementation: precompute smoothed
  BMO profile tables per θ_s bin, or smooth the final κ_pred radial
  kernel — cheapest is an extra convolution in `ClusterField`-style
  profile space for ALL halos.)
- Re-run the noiseless DC2 variant. Success criterion: A_matched = 1
  within ~10%. Outcome either validates the model amplitude at
  map scales (and pins the DC2 0.45 on resolution), or exposes a real
  amplitude bias — either way, a number for the referee.
- Bonus at zero cost: the residual (1 − A_matched) at LARGE scales
  isolates the missing 2-halo term (TODO 5.3), since pixel smoothing
  only affects sub-arcmin scales.

## Phase 1 — MICECATv2 access + extraction (blocked on CosmoHub account)

- **Action (Peter):** CosmoHub (cosmohub.pic.es) login — DES membership
  qualifies; request MICECATv2 access if not already granted.
- Custom query, ~12–20 disjoint patches of ~4 deg² across the octant
  (mirrors the 6-pixel DC2 design; more regions = per-region jackknife):
  columns `ra_gal, dec_gal, z_cgal, kappa, gamma1, gamma2,
  des_asahi_full_i_true / g,r,i,z evolution-corrected DES mags,
  lmstellar, lmhalo, flag_central, halo_id`. Depth cut at the DES
  z-band analysis limit (22.5) server-side; expect ~25k/deg² → ~2M rows
  per patch bundle, a few GB total (parquet/csv.bz2 download).
- Transfer to `/pscratch/sd/n/nugent/snkappa/micedata/` (repo m2218).

## Phase 2 — adapt the mock pipeline (~1 session)

- `scripts/mock_extract_mice.py`: same three products as DC2
  (foreground catalog, source sightlines carrying `kappa`, central-halo
  cluster catalog ≥6e13). Watch MICE mag conventions (evolution
  correction already applied in the `*_true` columns; pick the
  observed-frame set consistently).
- Two estimator routes in `mock_calibration.py`:
  (a) truth-based MockEstimator (as DC2) — isolates the halo chain;
  (b) NEW: color-based Taylor2011 on MICE DES griz — tests the real
  photometric estimator end-to-end (minus W1, which MICE lacks; the
  W1-share is bounded separately by the DES w1only variant).
- Port `--smooth-arcmin 0.85` from Phase 0 for the matched comparison.

## Phase 3 — MICE runs (debug queue, variants in parallel; ~1 session)

Same trio as DC2 (noiseless / fiducial / naive-SMHM) plus:
- resolution-matched noiseless (the headline number),
- estimator-(b) fiducial,
- cluster-tier-off (sizes the cluster share of the signal, feeding the
  cluster-mass-convention systematic in Table sys of the paper),
- aperture ladder (5′/10′/20′) to size the 2-halo deficit vs scale.
Deliverables: A_matched(MICE), λ_mock(MICE) vs 0.68±0.09 (DC2),
SMHM-inversion cost in a second universe, 2-halo correction curve.

## Phase 4 — absolute one-halo closure against DATA (parallel track; 1–2 sessions)

The resolution-free absolute test never comes from these lightcones.
Instead: predict the stacked excess surface density ΔΣ(R; M*) from our
exact chain (Nir1um-calibrated M* → posterior <Mh|M*> → capped BMO) for
lens samples matching published galaxy–galaxy lensing measurements
(SDSS Mandelbaum et al.; HSC; DES Y3 lens samples), and compare
amplitudes at 0.1–1 Mpc. This closes the one-halo amplitude with real
data and complements TODO 5.2 (DES Y3 mass-map cross-check, which is
again map-resolution-limited).

## Phase 5 — stretch options (only if Phases 0–4 leave a gap)

- AbacusSummit halo lightcones + high-res kappa maps (Nside 16384,
  0.21′) with an HOD/SHAM galaxy painting — true sub-arcmin mock truth,
  but a substantially bigger build.
- MillenniumTNG full-sky ray-tracing maps (resolution-convergence
  studied explicitly) — same galaxy-painting caveat.

## Success criteria / paper integration

- A_matched(DC2) and A_matched(MICE) consistent with 1 → "the halo
  model reproduces simulation convergence at map resolution" sentence +
  referee-response ammunition; λ_mock from two independent lightcones
  brackets the DES attenuation correction.
- ΔΣ closure → replaces the "bounds the halo-mass normalization at
  ~35%" statement with a direct one-halo amplitude validation.
- All new numbers land in the attenuation paragraph + Table sys of
  `paper/des_lensing_apjl.tex`; scripts and summaries tracked as for
  DC2 (`output/mock_mice/mock_summary*.json`).

## Logistics summary

| item | where | est. |
|---|---|---|
| Phase 0 smoothing run | local (or debug node) | ~1 h compute |
| CosmoHub query+download | cosmohub.pic.es → pscratch | few GB, manual gate |
| MICE extraction | Perlmutter login | minutes |
| MICE runs (6+ variants) | Perlmutter debug, 1 node | ~15 min |
| ΔΣ closure | local | analytic + tables |
