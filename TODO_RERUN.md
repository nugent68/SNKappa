# SNKappa — rerun handoff (2026-07-10, follow-up to TODO_REVIEW.md)

Audience: a session running in this repo on a machine that HAS the TAP
cache (`cache/` with the four regional parquet sets; e.g. the Perlmutter
checkout used for P5). The laptop that produced these commits could not
reach Data Lab (504 on `datalab.noirlab.edu/tap/sync`, 2026-07-10) and its
`cache/` was lost, so the reruns are handed off here.

Prereq: `git pull` — the code changes below are committed on `main`.
Tests: full suite passes (31 passed) on a fresh py3.14 venv
(`python3 -m venv .venv && .venv/bin/pip install -e ".[test]"`).

---

## What was changed (already committed — do NOT redo, just rerun)

1. **Fig. 3 caption fix (Overleaf clone, separate repo).** The caption's
   jackknife range `[-2.36, -1.40]` (stale, pre-recalibration) was changed
   to `[-2.00, -1.18]`, matching the body text and
   `paper_stats.json:jackknife_range`. Committed in the Overleaf clone as
   `1a01829` on the laptop — NOT yet pushed to Overleaf. Either push from
   the laptop or re-apply the one-line edit (line ~290 of
   `des_lensing_apjl.tex`).

2. **Cluster tier now participates in the host-plane excision**
   (`snkappa/batch.py`). `ClusterField.set_zsrc(cosmo, z_src,
   excise_frac=None)` drops clusters with `|z_cl - z_src| <
   excise_frac*(1+z_src)` from the foreground mask. Rationale: cluster
   catalog redshifts carry ~0.01–0.02 errors, so a cluster hosting the SN
   itself could pass the `z_cl < z_src - 0.02` cut and inject a large
   spurious kappa correlated with host environment — and clusters dominate
   the kappa_ext tail, so the published excision variant (galaxies only)
   did not actually test the strongest confounder channel.
   `scripts/des_full.py` passes `excise_frac` through in the `--excise-host`
   variant. Unit test: `tests/test_clusters.py::test_clusterfield_excision`.

3. **Per-SN cluster-proximity column** (`scripts/des_full.py`). Each SN row
   now gets `cl_dz_min_2am` = min |z_cl - z_SN|/(1+z_SN) over catalog
   clusters within 2 arcmin (NaN if none). `scripts/make_figs.py` adds a
   `no_cluster_at_sn_plane` robustness row: drop SNe with
   `cl_dz_min_2am < 0.05` and refit.

4. **Catalog band requirement relaxed** (`snkappa/catalog.py`).
   `clean_and_merge` now requires finite `mag_z` AND (finite `mag_w1` OR
   finite `mag_g`) instead of finite `mag_g` AND `mag_z`. The old cut
   silently dropped g-undetected massive red galaxies (g−z ≳ 3 at
   z ≤ 22.5) — the highest-M*/L kappa contributors — even though the
   default `nir1um_fsf` estimator needs only z+W1 (g is only for the
   Taylor2011 fallback). This UNDER-predicted kappa and biased A high.

5. **DESI crossmatch fiber collisions** (`snkappa/catalog.py`). When
   several fibers match one galaxy within 1", the CLOSEST now wins
   (previously last-write-wins in catalog row order). Tests:
   `tests/test_catalog.py` (two new end-to-end `clean_and_merge` tests).

---

## Why a full rerun is required

Change (4) alters the foreground catalog in every region, so **every
published number changes**: headline, all variant rows, Figs. 1–3,
Tables 1–3, paper_stats.json. Change (3) adds a column the new
`make_figs.py` row needs. Change (2) redefines the excise variant (by
design). Per TODO_REVIEW.md convention: complete ALL reruns, then update
every number in the paper in a single pass.

## Run list (in order; headline first warms nothing — cache already warm)

```
.venv/bin/python scripts/des_full.py                                  # headline
.venv/bin/python scripts/des_full.py --variant excise   --excise-host
.venv/bin/python scripts/des_full.py --variant nospecz  --no-specz
.venv/bin/python scripts/des_full.py --variant w1only   --require-w1
.venv/bin/python scripts/des_full.py --variant naive    --smhm-inverse naive
.venv/bin/python scripts/des_full.py --variant cap14.1  --logmh-max 14.1
.venv/bin/python scripts/des_full.py --variant mstar05  --mstar-offset 0.05
.venv/bin/python scripts/attenuation_mc.py           # TODO 2.1 artifact
.venv/bin/python scripts/make_figs.py                # stats + figures
```

Then:
- **Commit `output/des_full/attenuation.json`** (currently untracked
  anywhere — the paper's lambda numbers have no backing artifact) and
  reconcile the manuscript's lambda_gal = 0.90 vs TODO_REVIEW's 0.97
  against the fresh value.
- Update the manuscript on Overleaf in ONE pass: abstract, Sec. 4 numbers,
  Table 1 (top sightlines), Table 2 (all robustness rows + NEW
  `no_cluster_at_sn_plane` row), Table 3 (sys budget if the g-cut shifts
  A), Figs. 1–3, and the Method/Data text: state that the excision variant
  covers clusters, and that the catalog requires z + (W1 or g).
- Commit outputs + push both repos.

## Expected shifts / gates

- **g-cut relaxation**: adds W1-detected, g-undetected red galaxies →
  predicted kappa contrast up modestly → A likely down a little. This
  REMOVES a known upward bias on A; if |ΔA| > ~0.05, mention the change
  explicitly in the text (it is a bug fix, not a tuning).
- **excise variant (now incl. clusters)**: if the slope collapses here,
  the cluster-plane confounder is real and the headline needs rethinking —
  that is the point of the test. Expected: slope stable within errors,
  possibly slightly weaker (real signal from near-plane structure is also
  removed).
- **no_cluster_at_sn_plane row**: expect consistency with the headline.
- **closest-fiber fix**: negligible (sub-arcsec fiber collisions are rare);
  n_spec per region should shift by at most a handful.
- Permutation/block-bootstrap significance: regenerate via make_figs; if
  z_equiv lands between 3.5 and 3.7, consider 1e5 permutations so the
  quoted sigma is not floored at the 1e-4 resolution (cheap).

## Still open after this (unchanged from TODO_REVIEW P5 / review)

Latent-variable (errors-in-variables) regression as the headline upgrade;
2-halo term; 10^13.8–10^14 group-mass gap; DES Y3 mass-map cross-check;
DESI publication-policy sign-off for validation (v); shear + exact
magnification in the per-sightline prediction; SN coordinates instead of
host coordinates; R_IN < 3" with host excision by ID.
