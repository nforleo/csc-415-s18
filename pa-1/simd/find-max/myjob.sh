#!/bin/bash
#
#SBATCH --job-name=find-max
#SBATCH --error=error.txt
#SBATCH --output=output.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=5gb
#
#SBATCH --constraint=gpu

srun ./max 100
srun ./max 1000
srun ./max 10000
srun ./max 100000
srun ./max 1000000
srun ./max 10000000
srun ./max 100000000
srun ./max 1000000000
