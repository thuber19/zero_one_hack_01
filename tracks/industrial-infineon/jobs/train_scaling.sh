#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --job-name=process_seq_scaling
#SBATCH --output=slurm-scaling-%j.out
#SBATCH --error=slurm-scaling-%j.err

# Proxy
export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export HTTPS_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export http_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export https_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425

PROJECT_DIR="$HOME/process-sequence-model"
OUTPUT_BASE="$SCRATCH/process_seq_outputs"
cd "$PROJECT_DIR"

export RUN_COMMAND="$HOME/.pixi/bin/pixi run --manifest-path $PROJECT_DIR/pixi.toml"

# Train all 3 model sizes for scaling comparison
for SIZE in tiny small medium; do
    echo ""
    echo "=========================================="
    echo "  Training $SIZE model"
    echo "=========================================="
    OUTPUT_DIR="$OUTPUT_BASE/${SIZE}"
    mkdir -p "$OUTPUT_DIR"

    export OUTPUT_DIR="$OUTPUT_DIR"
    $RUN_COMMAND python3 src/train.py \
        --extra-data 10000 \
        --model-size "$SIZE" \
        --epochs 100 \
        --batch-size 128 \
        --lr 3e-4 \
        --device cuda \
        --seed 42
done

echo ""
echo "=== Scaling experiment complete ==="
echo "Results in: $OUTPUT_BASE/{tiny,small,medium}/"
