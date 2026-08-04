#!/bin/bash
#SBATCH -A m2218
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH -J zfix_mop
#SBATCH -o /pscratch/sd/n/nugent/lens/slurm-zfixmop-%j.out
srun --ntasks-per-node=1 --export=ALL /pscratch/sd/n/nugent/lens/node_runner_zfix_mop.sh
