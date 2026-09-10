#!/usr/bin/env bash
#SBATCH --job-name=cv_lam_extra
#SBATCH --account=cs-mlpath-2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-39%10
#SBATCH --output=../outputs/logs/cvlam_%A_%a.log

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source /mnt/scratch/users/plx526/camelyon/clam_env/bin/activate
cd /mnt/scratch/users/plx526/camelyon/code

# 4 lambdas x 10 folds = 40 tasks. config = task/10, fold = task%10
LAMBDAS=(0.01 0.02 0.15 0.2)
CFG=$(( SLURM_ARRAY_TASK_ID / 10 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 10 ))
LAM=${LAMBDAS[$CFG]}

echo "cvlam task ${SLURM_ARRAY_TASK_ID}: clam lambda=$LAM fold=$FOLD"
python train.py --model clam --lam $LAM --fold $FOLD --epochs 20