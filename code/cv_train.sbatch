#!/usr/bin/env bash
#SBATCH --job-name=cv_train
#SBATCH --account=cs-mlpath-2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-29%10
#SBATCH --output=../outputs/logs/cv_%A_%a.log

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source /mnt/scratch/users/plx526/camelyon/clam_env/bin/activate
cd /mnt/scratch/users/plx526/camelyon/code

# Map array index -> (fold, lambda)
LAMBDAS=(0.0 0.05 0.1)
FOLD=$(( SLURM_ARRAY_TASK_ID / 3 ))
LAM_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
LAM=${LAMBDAS[$LAM_IDX]}

echo "CV task ${SLURM_ARRAY_TASK_ID}: fold=$FOLD lambda=$LAM"
python train.py --fold $FOLD --lam $LAM --epochs 20