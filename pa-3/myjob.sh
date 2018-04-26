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
#
# below are two different ways to run your program, one is 
# running your program directly and the second is using a 
# profiler, please only run one of those at a time
#
## the following is just an example, you can add more calls
#srun ./prog
## the following runs a profiler on your program and sends 
## profiler information to the stderr (error.txt)
srun nvprof ./prog
