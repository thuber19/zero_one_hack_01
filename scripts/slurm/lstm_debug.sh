#!/bin/bash
#SBATCH --job-name=lstm_debug
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/lstm_debug_%j.out
#SBATCH --error=logs/lstm_debug_%j.err
#
# Smoke test: 1 epoch, batch=32, confirms import / data / checkpoint pipeline.
# Submit: sbatch scripts/slurm/lstm_debug.sh

set -euo pipefail
mkdir -p logs

module purge
module load python/3.11.7
module load cuda/12.2 || module load cuda

source "$HOME/zero_one_env/bin/activate"

REPO_DIR="${REPO_DIR:-$HOME/zero_one_hack_01}"
cd "$REPO_DIR"

DATA_DIR="$WORK/data/fab_sequences/shards"
OUT_DIR="$WORK/checkpoints/005-lstm-baseline"
mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

python src/train/train_lstm.py \
    --config configs/lstm_baseline.yaml \
    --data_dir "$DATA_DIR" \
    --tokenizer "$WORK/data/fab_sequences/tokenizer.json" \
    --test_sequences "$WORK/data/fab_sequences/test_sequences.json" \
    --output_dir "$OUT_DIR" \
    --debug

echo "[slurm] Debug run complete. Artifacts:"
ls -lh "$OUT_DIR"
