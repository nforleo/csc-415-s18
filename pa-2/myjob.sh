#!/bin/bash
#
#SBATCH --job-name=conv2d
#SBATCH --error=error.txt
#SBATCH --output=output.txt
#
## number of cores on the compute node
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#
#SBATCH --time=10:00
#
# necessary for running the AVX2 instructions
#SBATCH --constraint=gpu
#
module load LibTIFF
srun ./prog images/square.tif out.tif filters/gaussian_blur.txt seq 2
