#!/bin/bash
#
#SBATCH --job-name=filtering-images
#SBATCH --error=error.txt
#SBATCH --output=output.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=512mb
#
#SBATCH --constraint=gpu

srun ./prog images/square.tif out.tif filters/gaussian_blur.txt seq 2
