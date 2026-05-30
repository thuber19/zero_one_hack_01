#!/bin/bash
#SBATCH --job-name=procseq-sweep
#SBATCH --account=__ACCOUNT__
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=__QOS__
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --array=0-15
#SBATCH --output=logs/sweep_%A_%a.out

set -euo pipefail
module load python cuda || true
source "${VENV:-$HOME/procseq-venv}/bin/activate"
cd "${SOLUTION_DIR:-$SLURM_SUBMIT_DIR}"

SIZES=(tiny small base large)
DATA=(100 1000 5000 20000)
SIZE=${SIZES[$((SLURM_ARRAY_TASK_ID / 4))]}
DV=${DATA[$((SLURM_ARRAY_TASK_ID % 4))]}

python -m procseq.sweep_run --size "$SIZE" --data-per-family "$DV" \
  --seed 42 --out artifacts/sweep
