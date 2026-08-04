#!/bin/bash
#SBATCH -A m2218
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 06:00:00
#SBATCH -J kappa_zfix_prod
#SBATCH -o /pscratch/sd/n/nugent/lens/slurm-zfix-prod-%j.out
# All 6943 FB-ready kappa contributors, zfix mode.
# 6943 x ~330 s / 128 workers ~ 5 h.
srun --ntasks-per-node=1 /pscratch/sd/n/nugent/lens/node_runner_zfix.sh
