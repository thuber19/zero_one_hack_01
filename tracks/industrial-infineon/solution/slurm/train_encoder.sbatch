#!/bin/bash
#SBATCH --job-name=procseq-enc
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm-procseq-enc-%j.out
#SBATCH --error=slurm-procseq-enc-%j.err
#
# Train the procseq anomaly encoder (Task 3: anomaly + rule attribution +
# supervised-contrastive) on one Leonardo Booster A100.
#   cd tracks/industrial-infineon/solution && sbatch slurm/train_encoder.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-$HOME/procseq-venv}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
CONFIG="${CONFIG:-configs/leonardo_encoder.yaml}"

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A) started=$(date)"

python -m procseq.build_data --n-per-family 5000 --seed 42
srun accelerate launch --num_processes 1 -m procseq.train_encoder --config "$CONFIG"
python -m procseq.infer --task 3 --config "$CONFIG"
python -m procseq.run_eval --config "$CONFIG"
echo "=== DONE $(date) — see artifacts/metrics.json + artifacts/submission_task3.csv ==="
