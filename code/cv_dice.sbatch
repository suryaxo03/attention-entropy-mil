#!/usr/bin/env bash
#SBATCH --job-name=cv_dice
#SBATCH --account=cs-mlpath-2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --output=../outputs/logs/cv_dice_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=plx526@york.ac.uk

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source /mnt/scratch/users/plx526/camelyon/clam_env/bin/activate
cd /mnt/scratch/users/plx526/camelyon/code
python cv_dice_eval.py