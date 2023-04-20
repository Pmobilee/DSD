#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=CIN_Distillation
#SBATCH --cpus-per-task=18
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# Execute Program
module load 2022
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
# module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
source $HOME/.bashrc
conda activate D-SD

cd $Home/thesis/Diffusion_Thesis/cin_256
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/D-SD/lib
wandb login 4baa24c4fc6c8eed782cacb721d34977149d4fcb
python distill.py -t DSDI -n cin_2e9

