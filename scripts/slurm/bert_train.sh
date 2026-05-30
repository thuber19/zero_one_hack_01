#!/bin/bash
#SBATCH --job-name=bert_mlm_train
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/bert_train_%j.out
#SBATCH --error=logs/bert_train_%j.err

set -euo pipefail

module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0

source "$HOME/zero_one_env/bin/activate"

mkdir -p logs

# Stage data to node-local SSD
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
    --seed 42

# Rank 0 only: calibration and evaluation (torchrun sets SLURM_PROCID via LOCAL_RANK)
# Use SLURM_PROCID to guard rank 0 post-training steps
if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
  echo "Rank 0: running threshold calibration ..."
  python scripts/calibrate_threshold.py \
    --checkpoint "$WORK/checkpoints/002/best_model.pt" \
    --splits "$WORK/data/fab_sequences/splits.json" \
    --data-dir "$TMPDIR/fab_sequences" \
    --output "$WORK/checkpoints/002/threshold.json"

  echo "Rank 0: running evaluation ..."
  python src/eval_mlm.py \
    --checkpoint "$WORK/checkpoints/002/best_model.pt" \
    --threshold "$WORK/checkpoints/002/threshold.json" \
    --splits "$WORK/data/fab_sequences/splits.json" \
    --data-dir "$TMPDIR/fab_sequences" \
    --rules tracks/industrial-infineon/training_data/generation_rules.md \
    --output-dir results/002/
fi

echo "Training pipeline complete."
