#!/bin/bash
#SBATCH --job-name=bert_mlm_debug
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=logs/bert_debug_%j.out
#SBATCH --error=logs/bert_debug_%j.err

set -euo pipefail

module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0

source "$HOME/zero_one_env/bin/activate"

mkdir -p logs

# Stage data to node-local SSD to avoid repeated Lustre metadata hits
mkdir -p "$TMPDIR/fab_sequences"
cp -r "$WORK/data/fab_sequences/"*.csv "$TMPDIR/fab_sequences/" 2>/dev/null || true
cp "$WORK/data/fab_sequences/splits.json" "$TMPDIR/fab_sequences/" 2>/dev/null || true

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

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
