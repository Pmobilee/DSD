#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=Cuda_Test
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --time=00:05:00

# Execute Program

module load 2022
 
source activate D-SD

srun python distill.py -t SI 
