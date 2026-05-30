#!/bin/bash
#SBATCH --job-name=bert_mlm_train
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=480GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/bert_train_%j.out
#SBATCH --error=logs/bert_train_%j.err

set -euo pipefail

module purge
module load python/3.11.7
module load cuda/12.2 || module load cuda

source "$HOME/zero_one_env/bin/activate"

REPO_DIR="${REPO_DIR:-$HOME/zero_one_hack_01}"
cd "$REPO_DIR"

mkdir -p logs

# Stage data to node-local SSD
mkdir -p "$TMPDIR/fab_sequences"
cp -r "$WORK/data/fab_sequences/"*.csv "$TMPDIR/fab_sequences/" 2>/dev/null || true
cp "$WORK/data/fab_sequences/splits.json" "$TMPDIR/fab_sequences/" 2>/dev/null || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export PYTHONUNBUFFERED=1

export HTTP_PROXY=http://proxy-user:5dd1d2bd@10.99.0.138:4225
export HTTPS_PROXY=http://proxy-user:5dd1d2bd@10.99.0.138:4225
export http_proxy=http://proxy-user:5dd1d2bd@10.99.0.138:4225
export https_proxy=http://proxy-user:5dd1d2bd@10.99.0.138:4225

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

OUT_DIR="$WORK/checkpoints/002"
mkdir -p "$OUT_DIR"

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  src/train_mlm.py \
    --config configs/002_mlm.yaml \
    --data-dir "$TMPDIR/fab_sequences" \
    --splits "$WORK/data/fab_sequences/splits.json" \
    --vocab "$WORK/artifacts/001/vocab.json" \
    --output-dir "$OUT_DIR" \
    --seed 42 \
    $RESUME_FLAG

# Post-training: calibration and evaluation (rank 0 only)
if [ "${LOCAL_RANK:-0}" -eq 0 ]; then
  echo "[rank0] Running threshold calibration ..."
  python scripts/calibrate_threshold.py \
    --checkpoint "$OUT_DIR/best_model.pt" \
    --splits "$WORK/data/fab_sequences/splits.json" \
    --data-dir "$TMPDIR/fab_sequences" \
    --output "$OUT_DIR/threshold.json"

  echo "[rank0] Running evaluation ..."
  python src/eval_mlm.py \
    --checkpoint "$OUT_DIR/best_model.pt" \
    --threshold "$OUT_DIR/threshold.json" \
    --splits "$WORK/data/fab_sequences/splits.json" \
    --data-dir "$TMPDIR/fab_sequences" \
    --rules tracks/industrial-infineon/training_data/generation_rules.md \
    --output-dir results/002/
fi

echo "Training pipeline complete. Artifacts in $OUT_DIR"
