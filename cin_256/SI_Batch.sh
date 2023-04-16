#!/bin/bash
#SBATCH -t 0:5:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --mem=30G
module load cuda11.7
conda activate D-SD
python distill.py -t SI