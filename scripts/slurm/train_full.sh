#!/bin/bash
#SBATCH --job-name=gpt_fab_full
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=480GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Submit:
#   sbatch scripts/slurm/train_full.sh
#
# Resume:
#   RESUME=1 sbatch scripts/slurm/train_full.sh

set -euo pipefail
mkdir -p logs

module purge
module load python/3.11.7
module load cuda/12.2 || module load cuda

# Activate venv from provision.sh
source "$HOME/zero_one_env/bin/activate"

# One-shot data prep (idempotent — fast no-op if shards already exist)
REPO_DIR="${REPO_DIR:-$HOME/zero_one_hack_01}"
cd "$REPO_DIR"
DATA_OUT="$WORK/data/fab_sequences"
if [[ ! -f "$DATA_OUT/shards/val.pt" ]]; then
    echo "[slurm] Running prepare_data.py..."
    python scripts/prepare_data.py --config configs/train_gpt_fab.yaml --work_dir "$WORK"
fi

# Stage shards to node-local tmp (eliminate Lustre metadata pressure during training)
mkdir -p "$TMPDIR/shards"
cp "$DATA_OUT/shards/"*.pt "$TMPDIR/shards/"
cp "$DATA_OUT/tokenizer.json" "$TMPDIR/tokenizer.json"
cp "$DATA_OUT/test_sequences.json" "$TMPDIR/test_sequences.json"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export PYTHONUNBUFFERED=1

# Proxy required for internet access on compute nodes (Leonardo HPC)
export HTTP_PROXY=http://proxy-user:5dd1d2bd@10.99.0.138:4225
export HTTPS_PROXY=http://proxy-user:5dd1d2bd@10.99.0.138:4225
export http_proxy=http://proxy-user:5dd1d2bd@10.99.0.138:4225
export https_proxy=http://proxy-user:5dd1d2bd@10.99.0.138:4225

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

OUT_DIR="$WORK/checkpoints/001-gpt-fab"
mkdir -p "$OUT_DIR"

srun torchrun \
    --nnodes=1 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    src/train/train.py \
    --config configs/train_gpt_fab.yaml \
    --data_dir "$TMPDIR/shards" \
    --tokenizer "$TMPDIR/tokenizer.json" \
    --test_sequences "$TMPDIR/test_sequences.json" \
    --output_dir "$OUT_DIR" \
    $RESUME_FLAG

echo "[slurm] Training complete. Artifacts in $OUT_DIR"
ls -la "$OUT_DIR"
