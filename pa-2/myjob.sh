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
#
module load LibTIFF
## the following is just an example, you can add more calls
srun ./prog images/square.tif out.tif filters/gaussian-blur.txt par 16
