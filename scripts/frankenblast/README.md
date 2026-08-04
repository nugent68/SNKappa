# FrankenBlast 8-band campaign (deep-NIR κ contributors)

SED-fitting side of the deep-NIR mass upgrade: independent stellar
masses for the galaxies that dominate the predicted convergence in the
VIDEO / UltraVISTA fields, used to test the SNKappa estimator's
massive-end mass scale.

These files run **inside a FrankenBlast checkout** (they import
`run_lens_catalog`, `fit_host_sed`, `sbi_pp`), not inside this package.
On Perlmutter they live in
`/pscratch/sd/n/nugent/lens/frankenblast-host/` with the launcher
scripts one level up; they are tracked here so the campaign is
reproducible.

## Pipeline

1. `scripts/fb_targets.py` (this repo, runs locally) — ranks every
   aperture galaxy by its individual κ contribution using the pipeline's
   own halo tables, and writes `data_nir/fb_targets_*.csv` with
   Tractor-style photometry: DECam g/r/z + VISTA J/H/Ks (dereddened,
   `mw_transmission` = 1) + WISE W1/W2, plus `z_best` (DESI spec-z where
   available, else the DECaLS photo-z the κ pipeline uses).
2. `run_kappa_catalog.py` — 8-band BANDMAP over the campaign's
   `run_lens_catalog` machinery. Env knobs: `KFB_TMAX_PER_ITER`,
   `KFB_ERR_FLOOR`, `KFB_NPOSTERIOR`.
3. `run_kappa_zfix.py` — **the mode to use.** Same bands, but the
   redshift-conditioned model (`SBI_model_zfix_GPD2W_global.pt`) with
   `obs["redshift"]` set from `z_best`.
4. `node_runner_zfix.sh` + `run_zfix_prod.sh` — 64 workers/node.
5. `run_prospector_catalog.py` — direct prospector-alpha/dynesty fits
   (no emulator) as an independent check. `PROS_FIXZ=1` (default) pins
   `zred`; ~10x slower than SBI.

## Calibration results (2026-08-04)

| mode | yield | median runtime |
|---|---|---|
| z-free SBI (`run_kappa_catalog`) | ~17% | 666 s (83% hit the cap) |
| **zfix SBI (`run_kappa_zfix`)** | **100% (31/31)** | **300 s** |
| direct prospector, z pinned | 100% | ~3300 s |

Findings worth not rediscovering:

- **`run_lens_catalog.py` hardcodes `obs["redshift"] = None`**, so every
  fit went through the z-free branch even though the campaign ships a
  trained redshift-conditioned model. Its training set has
  `phot` shape (2e6, 35) = 17 fluxes + 17 uncertainties + 1 redshift.
  Supplying the redshift is what makes bright galaxies tractable: 7
  targets that never finished in 3.9 h z-free complete in ~5 min zfix.
- The z-free failures are **not** a walltime problem. `sbi_pp.py`'s MC
  loop does not increment its counter when a sampling iteration times
  out, so an out-of-distribution conditioning vector retries forever.
- **Do not inflate photometric errors to force convergence.** A floor
  sweep moved one galaxy's logM* 11.37 → 11.09 → 10.34 at 10/15/20%:
  the posterior slides to the prior. Measured aperture systematics
  (LS total vs VIDEO AUTO) are only 0.10–0.19 mag.
- Every VISTA measurement is out-of-distribution for the training noise
  model, which uses real 2MASS SNR curves (VIDEO is ~5 mag deeper);
  100% of targets exceed the 2MASS SNR at their magnitude by >10x. The
  machinery's resampling absorbs this for ordinary galaxies. A retrain
  with VISTA filters/depths is the clean fix.
- Each fit uses ~1 core and ~3 GB regardless of thread settings, so
  many thin workers beat few wide ones.
- Prospector: pin `zred` by **rebuilding the model from
  `model.config_dict`** — mutating `free_params` on a live
  `PolySpecModel` is a silent no-op. z-free dynesty fits hit `maxcall`,
  collapse to delta-function posteriors, and miss a known spec-z
  (0.225 → 0.377).

## Running it: use the debug queue

The regular queue gave a 2-node allocation that had not started after
an hour; the debug queue (max 30 min, 8 nodes, 2 concurrent jobs)
started within ~2 minutes and swept 94% of 6,943 galaxies in about an
hour of wall time. `node_runner_zfix_chunk.sh` bounds each job to a
global row range (`KFB_START`/`KFB_END`) so the list can be split
across several short jobs; `build_todo.py` + `zfix_mop.sh` then re-run
whatever no chunk finished, with the longest cap a 30-minute wall
allows (1700 s).

Sweep result: 6,552/6,943 fitted, median 369 s, p90 441 s.

Two operational gotchas:

- The shell fallback that logs a hard timeout (`echo ... >> $sumcsv`)
  creates the file WITHOUT a header if it fires first, so a naive
  `csv.DictReader` silently reads that galaxy as column names. The
  mop-up runner writes a header up front; readers should still sniff
  the first line (see `build_todo.py`).
- Do not poll job completion with `! squeue | grep -q <name>`: a
  transient empty `squeue` is indistinguishable from "job finished"
  and will report a running job as done with zero results. Poll
  `sacct` for a terminal state instead.

## Two-engine catalog: cross-calibrate before merging

Galaxies the emulator cannot fit even with z conditioned go to direct
prospector (`node_runner_pros_tier2.sh`). That makes the mass catalog
heterogeneous, and because the tier boundary tracks brightness (hence
kappa), any FB<->prospector offset would imprint as a spurious kappa
trend. `build_xcal.py` + `pros_xcal.sh` fit 60 FB-successful galaxies,
stratified in logM*, with BOTH engines so the offset can be measured,
tested for mass dependence, and corrected -- or reported as a
systematic. Do not merge the tiers without it.

## Caveat carried into the analysis

Fixing a photo-z rather than marginalizing over it understates σ(logM*)
by ~0.05–0.1 dex for the ~83% without spec-z (DECaLS σ_z ≈
0.02–0.05(1+z)). Refit a subsample at z ± σ_z to propagate.
