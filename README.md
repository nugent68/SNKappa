# SNKappa

External-convergence (κ_ext) estimator for a single strongly lensed supernova.

`snkappa` reconstructs the line-of-sight (LOS) mass distribution toward a strongly
lensed SN from a galaxy+halo catalog built by merging **DESI DR1 spectroscopy** with
**DESI Legacy Imaging Surveys DR10 (DECam) photometry and photometric redshifts**,
all retrieved through the public **NOIRLab Astro Data Lab** TAP service (no
credentials, no observatory-internal paths). It delivers the corrections a
strong-lens model needs:

- **Magnification**: `μ_true = μ_model / (1 − κ_ext)²` — the multiplicative flux
  correction `(1 − κ_ext)²` and magnitude offset `Δm = +2.5 log10[(1 − κ_ext)²]`.
- **Hubble constant** (if time delays are used): `H0_true = (1 − κ_ext) · H0_model`.

Both are reported as full probability distributions `P(κ_ext)`.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
# Step-0 availability check against the live Data Lab TAP schema:
.venv/bin/python -m snkappa check --config configs/sn2025wny.yaml
# Full pipeline:
.venv/bin/python -m snkappa run --config configs/sn2025wny.yaml
```

Outputs land in `output/<name>/`: a JSON summary, the merged LOS catalog
(FITS + parquet), diagnostic plots, and a data-version manifest with every query
string, row count, and retrieval timestamp.

## The κ_ext convention used here (read this before quoting numbers)

1. **What counts as "external":** every galaxy halo along the LOS *except* the
   primary deflector (and anything else inside `r_exclude`, default `5 × θ_E`) —
   those are assumed to be modeled explicitly by the strong-lens model. For
   SN 2025wny the default exclusion covers both G1 and G2.
2. **Group handling:** with `include_lens_group: false` (default), galaxies within
   `r_group_arcmin` of the deflector AND within `dz_group` of `z_lens` are also
   excluded (they are part of the deflector's environment that a group-aware
   strong-lens model would absorb). κ_ext is reported **both ways** as a diagnostic.
3. **Zero point (mass-sheet convention):** the raw halo sum is measured relative to
   the cosmic mean by subtracting the mean of the identical estimator run on
   `n_random_los` random sightlines in the same footprint:
   `κ_ext = κ_raw(SN) − ⟨κ_raw(random)⟩`. The scatter of the randoms provides the
   empirical correlated-LOS variance.
4. **Single-plane approximation:** each halo's convergence is computed at its own
   redshift with its own Σ_crit(z_l, z_s) and summed. This is the standard external-
   convergence sheet approximation, NOT multi-plane ray tracing (for a rigorous
   treatment couple the output catalog to e.g. `lenstronomy` multi-plane). The JSON
   output restates this caveat.
5. **Sign/physics sanity:** κ_ext > 0 (overdense LOS) means the lens model that
   ignored it *overestimates* H0 and *underestimates* the source magnification;
   the corrections above encode exactly this.

## Pipeline summary

1. **STEP 0 — availability check** (`snkappa check`, also run automatically before
   `run`): queries `TAP_SCHEMA` to confirm the exact tables/columns and prints a
   report. Verified 2026-07-02: `ls_dr10.tractor` ✅; `ls_dr10.photo_z` exists but
   is **south-only** → per-object fallback to `ls_dr9.photo_z` (identical `ls_id`
   in the north, where DR10 = DR9 BASS/MzLS); `desi_dr1.zpix` ✅ (no `OBJTYPE`
   column → quality cuts use `zcat_primary`, `zwarn=0`, `spectype='GALAXY'`,
   `coadd_fiberstatus=0`); photo-z tables serve **quantiles only** (median, ±68%,
   ±95%) → p(z) is reconstructed as a two-piece Gaussian pinned to the quantiles.
2. **Catalog**: LS tractor cone = master list (`brick_primary=1`, `type != 'PSF'`,
   `fracflux < 0.5`, `fracmasked < 0.4`, `fracin > 0.3`, bright-star/bad `maskbits`
   rejected; fluxes dereddened with `mw_transmission_*`). DESI `zpix` left-joined
   within 1″; spec-z wins over photo-z. Deflector (and optionally group) removed.
3. **Halos**: rest-frame color-based stellar mass (Taylor et al. 2011; g−i in the
   DECam south, g−z in the north where i-band is absent) → M_200c via a
   stellar-to-halo-mass relation (Behroozi et al. 2013 default, Moster et al. 2013
   option, configurable lognormal scatter) → c(M, z) (Diemer & Joyce 2019 via
   `colossus`) → truncated-NFW Σ(R) (Baltz, Marshall & Oguri 2009 profile with
   τ = r_t/r_s = c_200c, i.e. truncation at r_200c; pure NFW available).
4. **κ_ext**: κ_i = Σ_i(b_i)/Σ_crit(z_i, z_src) summed over LOS galaxies; photo-z
   galaxies marginalized over p(z) (with the z < z_src lensing-efficiency weight
   arising naturally — background galaxies contribute zero).
5. **Randoms**: mean-field zero point + empirical variance + the H0LiCOW-style
   weighted-number-count overdensity ζ = counts_LOS / median(counts_random)
   (unweighted, 1/r-weighted, and lensing-efficiency-weighted) as a
   halo-model-independent cross-check.
6. **Monte Carlo**: joint resampling of photo-z, SMHM scatter, M*/L scatter, and
   concentration scatter (`n_mc` draws) convolved with the random-LOS variance
   → P(κ_ext); percentiles 2.5/16/50/84/97.5 reported for both group branches.

## Reproducibility

- All queries cached in `cache/` keyed by (query, data release); reruns are
  network-free and deterministic (single seeded `numpy` RNG).
- `environment.yml` + `conda-lock.yml` pin the environment; the JSON output embeds
  the git commit, config hash, seed, and full query manifest.
- Data releases are pinned in the config (`ls_release: dr10`, `desi_release: dr1`);
  bump `desi_release` to `dr2` when NOIRLab serves it.

## Data acknowledgments (include in any publication using this tool)

- **DESI DR1**: DESI Data Release 1 (DESI Collaboration et al. 2025), CC BY 4.0.
  "This research used data obtained with the Dark Energy Spectroscopic Instrument
  (DESI). DESI construction and operations is managed by the Lawrence Berkeley
  National Laboratory. This material is based upon work supported by the U.S.
  Department of Energy, Office of Science, Office of High-Energy Physics, under
  Contract No. DE-AC02-05CH11231."
- **Legacy Surveys DR10**: "The Legacy Surveys consist of three individual and
  complementary projects: DECaLS, BASS, and MzLS... Full text at
  https://www.legacysurvey.org/acknowledgment/" (Dey et al. 2019).
- **Astro Data Lab**: "This research uses services or data provided by the Astro
  Data Lab, which is part of the Community Science and Data Center (CSDC) Program
  of NSF NOIRLab."

## References

Taylor et al. 2011, MNRAS 418, 1587 · Behroozi et al. 2013, ApJ 770, 57 ·
Moster et al. 2013, MNRAS 428, 3121 · Diemer & Joyce 2019, ApJ 871, 168 ·
Wright & Brainerd 2000, ApJ 534, 34 · Baltz, Marshall & Oguri 2009, JCAP 1, 15 ·
Rusu et al. 2017, MNRAS 467, 4220 (weighted number counts) ·
Falco, Gorenstein & Shapiro 1985, ApJ 289, L1 (mass-sheet degeneracy)
