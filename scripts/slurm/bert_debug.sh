#!/bin/bash
#SBATCH --job-name=bert_mlm_debug
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=480GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/bert_debug_%j.out
#SBATCH --error=logs/bert_debug_%j.err

set -euo pipefail

module purge
module load python/3.11.7
module load cuda/12.2 || module load cuda

source "$HOME/zero_one_env/bin/activate"

REPO_DIR="${REPO_DIR:-$HOME/zero_one_hack_01}"
cd "$REPO_DIR"

mkdir -p logs

# Stage data to node-local SSD to avoid repeated Lustre metadata hits
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
    --output-dir "$WORK/checkpoints/002" \
    --max-epochs 2 \
    --debug

echo "Debug run complete. Check $WORK/checkpoints/002/ for checkpoints."
