#!/bin/bash
#SBATCH --job-name=gpt_fab_eval
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --reservation=s_tra_ncc
#
# Recover eval_report.json from a completed (or crashed-at-eval) training job.
# Loads checkpoint_best.pt, runs test eval + memorization probe on 1 GPU.
#
# Submit:
#   sbatch scripts/slurm/eval_standalone.sh
#
# Override checkpoint:
#   CKPT=$WORK/checkpoints/001-gpt-fab/checkpoint_final.pt sbatch scripts/slurm/eval_standalone.sh

set -euo pipefail
mkdir -p logs

module purge
module load python/3.11.7
module load cuda/12.2 || module load cuda

source "$HOME/zero_one_env/bin/activate"

REPO_DIR="${REPO_DIR:-$HOME/zero_one_hack_01}"
cd "$REPO_DIR"

CKPT="${CKPT:-$WORK/checkpoints/001-gpt-fab/checkpoint_best.pt}"
DATA_DIR="$WORK/data/fab_sequences"

echo "[eval_standalone] checkpoint=$CKPT"
echo "[eval_standalone] data_dir=$DATA_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

python scripts/eval_standalone.py \
    --checkpoint "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --config     configs/train_gpt_fab.yaml

echo "[eval_standalone] Done."
