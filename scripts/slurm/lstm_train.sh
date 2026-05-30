#!/bin/bash
#SBATCH --job-name=lstm_train
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/lstm_train_%j.out
#SBATCH --error=logs/lstm_train_%j.err
#
# Full production run: 20 epochs, 2h budget, single A100.
# Submit:  sbatch scripts/slurm/lstm_train.sh
# Resume:  RESUME=1 sbatch scripts/slurm/lstm_train.sh

set -euo pipefail
mkdir -p logs

module purge
module load python/3.11.7
module load cuda/12.2 || module load cuda

source "$HOME/zero_one_env/bin/activate"

REPO_DIR="${REPO_DIR:-$HOME/zero_one_hack_01}"
cd "$REPO_DIR"

# Stage shards to node-local SSD (avoids Lustre metadata pressure during training)
mkdir -p "$TMPDIR/shards"
cp "$WORK/data/fab_sequences/shards/"*.pt "$TMPDIR/shards/"

OUT_DIR="$WORK/checkpoints/005-lstm-baseline"
mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

python src/train/train_lstm.py \
    --config configs/lstm_baseline.yaml \
    --data_dir "$TMPDIR/shards" \
    --tokenizer "$WORK/data/fab_sequences/tokenizer.json" \
    --test_sequences "$WORK/data/fab_sequences/test_sequences.json" \
    --output_dir "$OUT_DIR" \
    $RESUME_FLAG

echo "[slurm] Training complete. Artifacts in $OUT_DIR:"
ls -lh "$OUT_DIR"
