#!/bin/bash
#
#SBATCH --job-name=example-simd
#SBATCH --error=error.txt
#SBATCH --output=output.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=50mb
#
#SBATCH --constraint=gpu

srun ./prog
