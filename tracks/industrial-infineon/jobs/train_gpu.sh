#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --job-name=process_seq_gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e

PROJECT_DIR="$HOME/process-sequence-model"
export OUTPUT_DIR="$SCRATCH/process_seq_outputs"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

export RUN_COMMAND="$HOME/.pixi/bin/pixi run --as-is --manifest-path $PROJECT_DIR/pixi.toml env PYTHONUNBUFFERED=1"

echo "=== Starting training ==="
echo "Project: $PROJECT_DIR"
echo "Output:  $OUTPUT_DIR"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA:    $($RUN_COMMAND python3 -c 'import torch; print(f"available={torch.cuda.is_available()}, version={torch.version.cuda}")' 2>/dev/null || echo 'N/A')"
echo "Memory:  $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Step 1: Generate data
$RUN_COMMAND python3 src/generate_data.py \
    --extra-data 5000 \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

# Step 2: Train RF + Transformer (auto-detects GPU)
$RUN_COMMAND python3 src/train.py \
    --model-size small \
    --epochs 50 \
    --batch-size 32 \
    --lr 3e-4 \
    --seed 42

# Step 3: Self-evaluation
$RUN_COMMAND python3 src/evaluate.py \
    --self-eval \
    --output-dir "$OUTPUT_DIR" \
    --model-size small

# Step 4: Generate plots
$RUN_COMMAND python3 src/plot_results.py \
    --output-dir "$OUTPUT_DIR"

echo "=== Training + Evaluation complete ==="
echo "Outputs saved to: $OUTPUT_DIR"
