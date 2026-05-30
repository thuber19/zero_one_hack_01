#!/bin/bash
#
# Interactive training job launcher for Leonardo.
# Usage: bash jobs/run.sh
#

echo "============================================"
echo "  Process Sequence Model — Job Launcher"
echo "============================================"
echo ""

# Architecture
echo "Select architecture:"
echo "  1) transformer-tiny"
echo "  2) transformer-small"
echo "  3) transformer-medium"
echo "  4) lstm-tiny"
echo "  5) lstm-small"
echo "  6) lstm-medium"
echo ""
read -p "Choice [1-6]: " ARCH_CHOICE

case $ARCH_CHOICE in
    1) ARCH="transformer"; SIZE="tiny" ;;
    2) ARCH="transformer"; SIZE="small" ;;
    3) ARCH="transformer"; SIZE="medium" ;;
    4) ARCH="lstm"; SIZE="tiny" ;;
    5) ARCH="lstm"; SIZE="small" ;;
    6) ARCH="lstm"; SIZE="medium" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

# Data
echo ""
read -p "Sequences per family (e.g. 5000, 20000, 50000): " DATA

# Epochs
echo ""
read -p "Epochs [100]: " EPOCHS
EPOCHS="${EPOCHS:-100}"

# Batch size
echo ""
read -p "Batch size [512]: " BATCH
BATCH="${BATCH:-512}"

# Random Forest
echo ""
echo "Use Random Forest candidate filtering?"
echo "  1) Yes (default)"
echo "  2) No"
read -p "Choice [1-2]: " RF_CHOICE
RF_CHOICE="${RF_CHOICE:-1}"
if [ "$RF_CHOICE" = "2" ]; then
    RF_FLAG="--no-rf"
    RF_LABEL="OFF"
else
    RF_FLAG=""
    RF_LABEL="ON"
fi

# Physics
echo ""
echo "Use Physics refinery (for eval only)?"
echo "  1) No (default)"
echo "  2) Yes"
read -p "Choice [1-2]: " PHYS_CHOICE
PHYS_CHOICE="${PHYS_CHOICE:-1}"
if [ "$PHYS_CHOICE" = "2" ]; then
    PHYS_FLAG="--physics"
    PHYS_LABEL="ON"
else
    PHYS_FLAG=""
    PHYS_LABEL="OFF"
fi

# Fixed settings
GPUS=4
MEM=$((120 * GPUS))
CPUS=$((8 * GPUS))
OUTPUT_NAME="${ARCH}_${SIZE}_e${EPOCHS}_d${DATA}_rf${RF_LABEL}_phys${PHYS_LABEL}"
JOB_NAME="${ARCH}-${SIZE}-e${EPOCHS}"

echo ""
echo "============================================"
echo "  Submitting:"
echo "    Arch:     $ARCH $SIZE"
echo "    Epochs:   $EPOCHS (early stopping patience=20)"
echo "    Data:     ${DATA}/family ($(( DATA * 3 )) total)"
echo "    Batch:    $BATCH"
echo "    RF:       $RF_LABEL"
echo "    Physics:  $PHYS_LABEL"
echo "    GPUs:     $GPUS (${MEM}GB RAM, ${CPUS} CPUs)"
echo "    Output:   \$SCRATCH/runs/$OUTPUT_NAME"
echo "============================================"
echo ""
read -p "Submit? [Y/n]: " CONFIRM
CONFIRM="${CONFIRM:-Y}"
if [[ "$CONFIRM" != "Y" && "$CONFIRM" != "y" ]]; then
    echo "Cancelled."
    exit 0
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=${GPUS}
#SBATCH --mem=${MEM}GB
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=3:00:00
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=slurm-${OUTPUT_NAME}-%j.out
#SBATCH --error=slurm-${OUTPUT_NAME}-%j.err

set -e

PROJECT_DIR="\$HOME/process-sequence-model"
export OUTPUT_DIR="\$SCRATCH/runs/${OUTPUT_NAME}"
export PYTHONUNBUFFERED=1
mkdir -p "\$OUTPUT_DIR"
cd "\$PROJECT_DIR"

export RUN="\$HOME/.pixi/bin/pixi run --as-is --manifest-path \$PROJECT_DIR/pixi.toml env PYTHONUNBUFFERED=1"

echo "============================================"
echo "  ${ARCH} ${SIZE} | ${EPOCHS} epochs | ${DATA}/family"
echo "  RF: ${RF_LABEL} | Physics: ${PHYS_LABEL}"
echo "  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Started: \$(date)"
echo "============================================"
echo ""

# Step 1: Generate data
\$RUN python3 src/generate_data.py \\
    --extra-data ${DATA} \\
    --output-dir "\$OUTPUT_DIR" \\
    --seed 42

# Step 2: Train model (+ RF unless disabled)
\$RUN python3 src/train.py \\
    --arch ${ARCH} \\
    --model-size ${SIZE} \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH} \\
    --lr 3e-4 \\
    --seed 42 ${RF_FLAG}

# Step 3: Evaluate
\$RUN python3 src/evaluate.py \\
    --self-eval \\
    --output-dir "\$OUTPUT_DIR" \\
    --model-size ${SIZE} ${RF_FLAG} ${PHYS_FLAG}

# Step 4: Plots
\$RUN python3 src/plot_results.py \\
    --output-dir "\$OUTPUT_DIR"

echo ""
echo "=== DONE — \$(date) ==="
echo "Results in: \$OUTPUT_DIR"
EOF
