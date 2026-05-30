#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=480GB
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --job-name=medium_train
#SBATCH --output=slurm-medium-%j.out
#SBATCH --error=slurm-medium-%j.err

set -e

PROJECT_DIR="$HOME/process-sequence-model"
export OUTPUT_DIR="$SCRATCH/process_seq_medium"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

export RUN="$HOME/.pixi/bin/pixi run --as-is --manifest-path $PROJECT_DIR/pixi.toml env PYTHONUNBUFFERED=1"

echo "============================================"
echo "  MEDIUM MODEL — $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "============================================"

# Generate 20K per family (60K total) — fits in memory
$RUN python3 src/generate_data.py \
    --extra-data 50000 \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

# Train medium model (8 layers, 512 dim, ~20M params)
$RUN python3 src/train.py \
    --model-size medium \
    --epochs 100 \
    --batch-size 512 \
    --lr 3e-4 \
    --seed 42

# Evaluate
$RUN python3 src/evaluate.py \
    --self-eval \
    --output-dir "$OUTPUT_DIR" \
    --model-size medium

# Plots
$RUN python3 src/plot_results.py \
    --output-dir "$OUTPUT_DIR"

echo "=== DONE — $(date) ==="
echo "Results in: $OUTPUT_DIR"
