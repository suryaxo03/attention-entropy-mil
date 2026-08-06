#!/usr/bin/env bash
#SBATCH --job-name=lam_sweep
#SBATCH --account=cs-mlpath-2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-2
#SBATCH --output=../outputs/logs/sweep_%A_%a.log

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source /mnt/scratch/users/plx526/camelyon/clam_env/bin/activate
cd /mnt/scratch/users/plx526/camelyon/code

# lambda values indexed by array task
LAMBDAS=(0.15 0.2 0.3)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "Training fold 0 with lambda=$LAM"
python train.py --fold 0 --lam $LAM --epochs 20