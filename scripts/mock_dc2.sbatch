#!/bin/bash
#SBATCH -A m2218
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J snkappa-mock
#SBATCH -o mock_dc2_%j.log

# End-to-end cosmoDC2 mock calibration (TODO 5.1). The three variants are
# independent serial processes (~10 min each), run concurrently on one
# debug node. Run scripts/mock_extract_dc2.py first.

cd /pscratch/sd/n/nugent/snkappa
PIX="10066 10324 8793 9171 9430 9689"
PY=./venv/bin/python

$PY scripts/mock_calibration.py --pixels $PIX --datadir mockdata \
    --out output/mock_dc2 > mock_fiducial.log 2>&1 &

$PY scripts/mock_calibration.py --pixels $PIX --datadir mockdata \
    --out output/mock_dc2 --smhm-inverse naive --variant naive \
    > mock_naive.log 2>&1 &

$PY scripts/mock_calibration.py --pixels $PIX --datadir mockdata \
    --out output/mock_dc2 --sigz0 0.001 --outlier-frac 0 \
    --mstar-scatter 0.001 --clmass-scatter 0.02 --variant noiseless \
    > mock_noiseless.log 2>&1 &

wait
echo MOCK_ALL_DONE
