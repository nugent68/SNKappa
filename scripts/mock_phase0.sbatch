#!/bin/bash
#SBATCH -A m2218
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J snkappa-phase0
#SBATCH -o mock_phase0_%j.log

# Phase 0 of docs/MICE_PLAN.md: resolution-matched closure on cosmoDC2.
# Noiseless pipeline with the PREDICTION smoothed by a Gaussian kernel of
# width sigma, scanning sigma to bracket the effective smoothing of the
# cosmoDC2 lensing maps (Nside=4096 shells). Success = A_matched -> 1.

cd /pscratch/sd/n/nugent/snkappa
PIX="10066 10324 8793 9171 9430 9689"
PY=./venv/bin/python
NOISELESS="--sigz0 0.001 --outlier-frac 0 --mstar-scatter 0.001 --clmass-scatter 0.02"

for S in 0.25 0.50 0.85 1.30; do
  $PY scripts/mock_calibration.py --pixels $PIX --datadir mockdata \
      --out output/mock_dc2 $NOISELESS --smooth-arcmin $S \
      --variant sm$S > mock_sm$S.log 2>&1 &
done
wait
echo PHASE0_ALL_DONE
