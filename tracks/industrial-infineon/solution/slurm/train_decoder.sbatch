#!/bin/bash
#SBATCH --job-name=procseq-dec
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm-procseq-dec-%j.out
#SBATCH --error=slurm-procseq-dec-%j.err
#
# Train the procseq decoder (Tasks 1+2) on one Leonardo Booster A100.
# Submit from the solution dir, after running setup_leonardo.sh once:
#   cd tracks/industrial-infineon/solution && sbatch slurm/train_decoder.sbatch
# Single GPU + plain Accelerate (no DeepSpeed) — the models are small; DeepSpeed
# is only used by the scaling sweep. Override CONFIG=... to change preset.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-$HOME/procseq-venv}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
CONFIG="${CONFIG:-configs/leonardo_decoder.yaml}"

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A) started=$(date)"

# Build training/eval data + tokenizer first (no GPU needed, fast).
python -m procseq.build_data --n-per-family 5000 --seed 42

srun accelerate launch --num_processes 1 -m procseq.train_decoder --config "$CONFIG"

# Self-score on our labelled mirrors (gives metrics.json).
python -m procseq.infer --task 1 --config "$CONFIG"
python -m procseq.infer --task 2 --config "$CONFIG"
python -m procseq.run_eval --config "$CONFIG"
# Real submission from the organizers' data/eval_input_valid.csv (600 rows).
python -m procseq.infer --task 1 --real --config "$CONFIG"
python -m procseq.infer --task 2 --real --config "$CONFIG"
echo "=== DONE $(date) — metrics.json + submission_task{1,2}_real.csv ==="
