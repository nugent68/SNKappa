# DES-SN5YR input files

Both files are SNANA-format ("VARNAMES:/SN:") tables from the public
DES-SN5YR data release (Abbott et al. 2024, ApJL 973, L14; Sánchez et
al. 2024; Vincenzi et al. 2024), redistributed here unmodified so that
`scripts/des_full.py` runs from a fresh clone:

- `DES_HD.csv` — the Hubble diagram: CID, zHD, MU, MUERR, PROBIA_BEAMS
  (BEAMS Type-Ia probability), one row per SN after BBC bias corrections.
- `DES_meta.csv` — the per-SN metadata FITRES: FIELD assignment, host
  coordinates (HOST_RA/HOST_DEC), HOST_LOGMASS, light-curve parameters.

Upstream: https://github.com/des-science/DES-SN5YR (release paper data
products; the equivalent merged table there is
`4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv` at the pre-Dovekie
release commit `97d8d0c`, which however lacks the FIELD column carried
by the SNANA FITRES) and the DES-SN5YR Zenodo record.

Note (July 2025): the upstream `main` branch has since been updated to
the Dovekie calibration (`DES-Dovekie_HD.csv`); the analysis in the
paper uses the original DES-SN5YR release distances as archived here.
