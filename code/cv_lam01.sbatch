#!/usr/bin/env bash
#SBATCH --job-name=cv_lam01
#SBATCH --account=cs-mlpath-2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-9%10
#SBATCH --output=../outputs/logs/cvlam01_%A_%a.log
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source /mnt/scratch/users/plx526/camelyon/clam_env/bin/activate
cd /mnt/scratch/users/plx526/camelyon/code
python train.py --model clam --lam 0.1 --fold $SLURM_ARRAY_TASK_ID --epochs 20