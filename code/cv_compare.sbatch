#!/usr/bin/env bash
#SBATCH --job-name=cv_compare
#SBATCH --account=cs-mlpath-2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-59%10
#SBATCH --output=../outputs/logs/compare_%A_%a.log

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source /mnt/scratch/users/plx526/camelyon/clam_env/bin/activate
cd /mnt/scratch/users/plx526/camelyon/code

# 6 configs x 10 folds = 60 tasks
# config index = task / 10 ; fold = task % 10
CONFIGS_MODEL=(mean max abmil transmil clam clam)
CONFIGS_LAM=(0.0 0.0 0.0 0.0 0.0 0.05)

CFG=$(( SLURM_ARRAY_TASK_ID / 10 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 10 ))
MODEL=${CONFIGS_MODEL[$CFG]}
LAM=${CONFIGS_LAM[$CFG]}

echo "compare task ${SLURM_ARRAY_TASK_ID}: model=$MODEL lambda=$LAM fold=$FOLD"
python train.py --model $MODEL --lam $LAM --fold $FOLD --epochs 20