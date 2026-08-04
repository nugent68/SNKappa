#!/bin/bash
#SBATCH -A m2218
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -J pros_xcal
#SBATCH -o /pscratch/sd/n/nugent/lens/slurm-prosxcal-%j.out
export KFB_CSV=/pscratch/sd/n/nugent/lens/fb_targets_xcal.csv
export PROS_TAG=xcal PROS_CAP=9000 WPN=48
srun --ntasks-per-node=1 --export=ALL /pscratch/sd/n/nugent/lens/node_runner_pros_tier2.sh
