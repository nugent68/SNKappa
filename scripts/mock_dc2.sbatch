#!/bin/bash
#SBATCH -A m2218
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J snkappa-mock
#SBATCH -o mock_dc2_%j.log

# End-to-end cosmoDC2 mock calibration (TODO 5.1): fiducial, naive-SMHM,
# and noiseless variants. Run scripts/mock_extract_dc2.py first.

cd /pscratch/sd/n/nugent/snkappa
PIX="10066 10324 8793 9171 9430 9689"
PY=./venv/bin/python

$PY scripts/mock_calibration.py --pixels $PIX --datadir mockdata \
    --out output/mock_dc2

$PY scripts/mock_calibration.py --pixels $PIX --datadir mockdata \
    --out output/mock_dc2 --smhm-inverse naive --variant naive

$PY scripts/mock_calibration.py --pixels $PIX --datadir mockdata \
    --out output/mock_dc2 --sigz0 0.001 --outlier-frac 0 \
    --mstar-scatter 0.001 --clmass-scatter 0.02 --variant noiseless

echo MOCK_ALL_DONE
