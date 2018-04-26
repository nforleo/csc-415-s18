#!/bin/bash
#
#SBATCH --job-name=conv1d
#SBATCH --error=error.txt
#SBATCH --output=output.txt
#
## number of cores on the compute node
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#
#SBATCH --time=10:00
#
# necessary for running in the GPU
#SBATCH --constraint=gpu
#
#
module load GCC
module load CUDA
## the following is just an example, you can add more calls
srun ./prog
