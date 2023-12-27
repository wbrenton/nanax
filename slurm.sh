#!/bin/bash
#SBATCH --job-name=carperai
#SBATCH --partition=g40x
#SBATCH --account=carperai
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=12
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out

source venv/bin/activate

srun python nanax/vit/tune.py