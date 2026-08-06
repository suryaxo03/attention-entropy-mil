#!/usr/bin/env bash
# Load Viking's GPU-enabled PyTorch, then activate the project venv.
# Usage (after landing on a compute node):  source setup.sh
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source /mnt/scratch/users/plx526/camelyon/clam_env/bin/activate
echo "Environment ready. CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
