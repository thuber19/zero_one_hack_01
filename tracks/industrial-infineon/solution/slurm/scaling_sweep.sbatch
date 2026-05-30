#!/bin/bash
#SBATCH --job-name=procseq-sweep
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --array=0-15
#SBATCH --output=slurm-procseq-sweep-%A_%a.out
#SBATCH --error=slurm-procseq-sweep-%A_%a.err
#
# Scaling grid (size x data-volume), one array task per cell, one A100 each.
# Run build_data once first (sweep cells stage the eval mirrors from artifacts/):
#   cd tracks/industrial-infineon/solution
#   sbatch slurm/train_decoder.sbatch   # or: python -m procseq.build_data --n-per-family 20000
#   sbatch slurm/scaling_sweep.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-$HOME/procseq-venv}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

SIZES=(tiny small base large)
DATA=(100 1000 5000 20000)
SIZE=${SIZES[$((SLURM_ARRAY_TASK_ID / 4))]}
DV=${DATA[$((SLURM_ARRAY_TASK_ID % 4))]}

python -m procseq.sweep_run --size "$SIZE" --data-per-family "$DV" \
  --seed 42 --out artifacts/sweep --eval-from artifacts
