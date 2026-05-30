#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --job-name=overnight_train
#SBATCH --output=slurm-overnight-%j.out
#SBATCH --error=slurm-overnight-%j.err

set -e

PROJECT_DIR="$HOME/process-sequence-model"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"

export RUN="$HOME/.pixi/bin/pixi run --as-is --manifest-path $PROJECT_DIR/pixi.toml env PYTHONUNBUFFERED=1"

echo "============================================"
echo "  OVERNIGHT TRAINING RUN"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Started: $(date)"
echo "============================================"
echo ""

# ── Step 1: Generate large dataset (50K per family) ──
echo "=== Generating 50K sequences per family ==="
export OUTPUT_DIR="$SCRATCH/process_seq_overnight"
mkdir -p "$OUTPUT_DIR"

$RUN python3 src/generate_data.py \
    --extra-data 100000 \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

# ── Step 2: Train all 3 model sizes ──
for SIZE in tiny small medium; do
    echo ""
    echo "============================================"
    echo "  Training $SIZE model — $(date)"
    echo "============================================"

    SIZE_DIR="$OUTPUT_DIR/$SIZE"
    mkdir -p "$SIZE_DIR"

    # Copy tokenizer and data to size-specific dir
    cp "$OUTPUT_DIR/tokenizer.txt" "$SIZE_DIR/"
    cp "$OUTPUT_DIR/sequences.json" "$SIZE_DIR/"
    cp "$OUTPUT_DIR/transitions.json" "$SIZE_DIR/"

    # Train
    export OUTPUT_DIR="$SIZE_DIR"
    $RUN python3 src/train.py \
        --model-size "$SIZE" \
        --epochs 100 \
        --batch-size 128 \
        --lr 3e-4 \
        --seed 42

    # Evaluate
    $RUN python3 src/evaluate.py \
        --self-eval \
        --output-dir "$SIZE_DIR" \
        --model-size "$SIZE"

    # Plot individual results
    $RUN python3 src/plot_results.py \
        --output-dir "$SIZE_DIR"

    echo "  $SIZE model done — $(date)"
done

# ── Step 3: Scaling comparison plot ──
echo ""
echo "=== Generating scaling comparison ==="
export OUTPUT_DIR="$SCRATCH/process_seq_overnight"
$RUN python3 src/plot_results.py \
    --output-dir "$OUTPUT_DIR" \
    --scaling-dirs "$OUTPUT_DIR/tiny" "$OUTPUT_DIR/small" "$OUTPUT_DIR/medium"

echo ""
echo "============================================"
echo "  OVERNIGHT RUN COMPLETE — $(date)"
echo "  Results in: $SCRATCH/process_seq_overnight/"
echo "    tiny/   — training_history.json, eval_results/, plots/"
echo "    small/  — training_history.json, eval_results/, plots/"
echo "    medium/ — training_history.json, eval_results/, plots/"
echo "    plots/  — scaling_comparison.png"
echo "============================================"
