#!/bin/bash
#
# Interactive training job launcher for Leonardo.
# Usage: bash jobs/run.sh
#

# ---------------------------------------------------------------------------
# procseq pipeline (Claude's solution under solution/). Selected as options
# 7/8 in the architecture menu. Self-contained venv (~/procseq-venv), single
# A100, plain Accelerate. Produces self-eval metrics.json + the real organizer
# submissions (submission_task*_real.csv) in $SCRATCH/runs/<name>.
# ---------------------------------------------------------------------------
submit_procseq() {
    echo ""
    echo "=== procseq FULL pipeline (trains decoder + encoder, all 3 tasks) ==="
    read -p "Model size [tiny/small/base/large] (base): " PSIZE;  PSIZE="${PSIZE:-base}"
    read -p "Sequences per family [5000]: "               PDATA;  PDATA="${PDATA:-5000}"
    read -p "Max training steps per model [4000]: "       PSTEPS; PSTEPS="${PSTEPS:-4000}"
    read -p "Batch size [64]: "                           PBATCH; PBATCH="${PBATCH:-64}"

    local PROJECT_DIR="$HOME/process-sequence-model"
    local SOL="$PROJECT_DIR/solution"
    local PIXI="$HOME/.pixi/bin/pixi"
    local OUTNAME="procseq_full_${PSIZE}_d${PDATA}_s${PSTEPS}"
    local OUTDIR="${SCRATCH:-$HOME}/runs/${OUTNAME}"
    local CFG="${SOL}/configs/_run_full.yaml"

    if [ ! -d "$SOL" ]; then
        echo "ERROR: $SOL not found. Check out the branch with solution/ into $PROJECT_DIR."
        exit 1
    fi

    # Ensure the 'procseq' pixi env is present + up to date on the LOGIN node (compute
    # nodes have no internet). Idempotent: fast when nothing changed, re-solves only when
    # pixi.toml/lock changed (e.g. a torch version fix) -- so always run it.
    echo "Ensuring the 'procseq' pixi environment is up to date (login node)..."
    "$PIXI" install --manifest-path "$PROJECT_DIR/pixi.toml" -e procseq || { echo "pixi install failed"; exit 1; }

    mkdir -p "$OUTDIR"

    # Write the run config (login node).
    cat > "$CFG" <<YAML
run_name: ${OUTNAME}
seed: 42
precision: bf16
artifacts: ${OUTDIR}
decoder_ckpt: ${OUTDIR}/decoder
encoder_ckpt: ${OUTDIR}/encoder
decoder:
  size: ${PSIZE}
  max_len: 256
  data_per_family: ${PDATA}
  batch_size: ${PBATCH}
  lr: 0.0006
  max_steps: ${PSTEPS}
  eval_every: 500
  constrained_decode: true
encoder:
  size: ${PSIZE}
  max_len: 256
  data_per_family: ${PDATA}
  batch_size: ${PBATCH}
  lr: 0.0002
  max_steps: ${PSTEPS}
  warmup_frac: 0.1
  weight_decay: 0.01
  eval_every: 250
  contrastive:
    enabled: true
    weight: 0.5
    temperature: 0.1
YAML

    echo ""
    echo "============================================"
    echo "  procseq FULL | size=${PSIZE} | ${PDATA}/family | ${PSTEPS} steps/model | bs=${PBATCH}"
    echo "  trains decoder (T1+T2) + encoder (T3); infers all 3 tasks (pure + physics hybrid)"
    echo "  env:    pixi -e procseq"
    echo "  output: ${OUTDIR}"
    echo "============================================"
    read -p "Submit? [Y/n]: " C; C="${C:-Y}"
    [[ "$C" == "Y" || "$C" == "y" ]] || { echo "Cancelled."; exit 0; }

    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --job-name=procseq-full
#SBATCH --output=slurm-${OUTNAME}-%j.out
#SBATCH --error=slurm-${OUTNAME}-%j.err

set -e
RUN="${PIXI} run --as-is --manifest-path ${PROJECT_DIR}/pixi.toml -e procseq env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=${SOL} PROCSEQ_ARTIFACTS=${OUTDIR}"
cd "${SOL}"
echo "node=\$(hostname) gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',' || echo N/A) start=\$(date)"

# ONE call does everything: build data -> train decoder + encoder IN PARALLEL
# (auto-splits the 4 GPUs, 2 per model) -> infer all 3 tasks (pure + physics
# hybrids) -> self-eval -> OFFICIAL scores -> real submissions.
\$RUN python -m procseq.run_all --config "${CFG}" --parallel-train

echo "=== DONE \$(date) ==="
echo "live training logs: ${OUTDIR}/train_decoder.log , ${OUTDIR}/train_encoder.log"
echo "metrics:            ${OUTDIR}/metrics.json"
echo "real submissions:   ${OUTDIR}/submission_task*_real.csv (pure) + *_hybrid_real.csv (physics)"
EOF
    echo "Submitted. Watch with:  squeue --me   |   tail -f slurm-${OUTNAME}-*.out"
}

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
echo "  --- Claude's procseq pipeline (own pixi env, single A100) ---"
echo "  7) procseq-full      (trains decoder + encoder; all 3 tasks, pure + physics hybrid)"
echo ""
read -p "Choice [1-7]: " ARCH_CHOICE

case $ARCH_CHOICE in
    1) ARCH="transformer"; SIZE="tiny" ;;
    2) ARCH="transformer"; SIZE="small" ;;
    3) ARCH="transformer"; SIZE="medium" ;;
    4) ARCH="lstm"; SIZE="tiny" ;;
    5) ARCH="lstm"; SIZE="small" ;;
    6) ARCH="lstm"; SIZE="medium" ;;
    7) submit_procseq; exit 0 ;;
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
echo "Use Physics refinery?"
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

# Step 1: Generate data + eval CSVs
echo "=== Step 1: Generating data ==="
\$RUN python3 src/generate_data.py \\
    --extra-data ${DATA} \\
    --output-dir "\$OUTPUT_DIR" \\
    --eval-split 0.01 \\
    --seed 42

# Step 1b: Run baselines (fast, no GPU needed)
echo ""
echo "=== Step 1b: Baselines ==="
\$RUN python3 src/baseline.py \\
    --model-dir "\$OUTPUT_DIR" \\
    --eval-valid "\$OUTPUT_DIR/eval_input_valid.csv" \\
    --eval-anomaly "\$OUTPUT_DIR/eval_input_anomaly.csv" \\
    --out-dir "\$OUTPUT_DIR/baselines"

echo ""
echo "--- Baseline: Random ---"
\$RUN python3 data/eval_metrics.py \\
    --task next-step \\
    --ground-truth "\$OUTPUT_DIR/eval_set_valid.csv" \\
    --predictions "\$OUTPUT_DIR/baselines/random/nextstep.csv"

echo ""
echo "--- Baseline: Frequency (bigram) ---"
\$RUN python3 data/eval_metrics.py \\
    --task next-step \\
    --ground-truth "\$OUTPUT_DIR/eval_set_valid.csv" \\
    --predictions "\$OUTPUT_DIR/baselines/frequency/nextstep.csv"

# Step 2: Train model (+ RF unless disabled)
echo ""
echo "=== Step 2: Training ==="
\$RUN python3 src/train.py \\
    --arch ${ARCH} \\
    --model-size ${SIZE} \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH} \\
    --lr 3e-4 \\
    --seed 42 ${RF_FLAG}

# Step 3a: Inference on self-eval data
echo ""
echo "=== Step 3a: Inference (self-eval) ==="
\$RUN python3 src/inference.py \\
    --model-dir "\$OUTPUT_DIR" \\
    --eval-valid "\$OUTPUT_DIR/eval_input_valid.csv" \\
    --eval-anomaly "\$OUTPUT_DIR/eval_input_anomaly.csv" \\
    --out-dir "\$OUTPUT_DIR/self_eval_submissions" \\
    ${RF_FLAG} ${PHYS_FLAG}

# Step 3b: Inference on official eval data
echo ""
echo "=== Step 3b: Inference (official eval) ==="
\$RUN python3 src/inference.py \\
    --model-dir "\$OUTPUT_DIR" \\
    --eval-valid "\$PROJECT_DIR/data/eval_input_valid.csv" \\
    --eval-anomaly "\$PROJECT_DIR/data/eval_input_anomaly.csv" \\
    --out-dir "\$OUTPUT_DIR/official_submissions" \\
    ${RF_FLAG} ${PHYS_FLAG}

# Step 4: Score self-eval using official eval_metrics.py
echo ""
echo "=== Step 4: Self-Evaluation (official scorer) ==="
\$RUN python3 data/eval_metrics.py \\
    --task next-step \\
    --ground-truth "\$OUTPUT_DIR/eval_set_valid.csv" \\
    --predictions "\$OUTPUT_DIR/self_eval_submissions/nextstep.csv"

\$RUN python3 data/eval_metrics.py \\
    --task completion \\
    --ground-truth "\$OUTPUT_DIR/eval_set_valid.csv" \\
    --predictions "\$OUTPUT_DIR/self_eval_submissions/completion.csv"

\$RUN python3 data/eval_metrics.py \\
    --task anomaly \\
    --ground-truth "\$OUTPUT_DIR/eval_set_forbidden.csv" \\
    --predictions "\$OUTPUT_DIR/self_eval_submissions/anomaly.csv" \\
    --valid-supplement "\$OUTPUT_DIR/eval_set_valid_supplement.csv"

# Step 4b: Comprehensive anomaly stress-test (known-BAD dataset, ALL 10 rules)
echo ""
echo "=== Step 4b: Comprehensive anomaly stress-test (every forbidden rule) ==="
# Build a full-size, foolproof known-bad testset: every one of the 10 forbidden
# patterns is produced at least once and each label is independently re-verified
# by the reference checker (make_bad_testset.py also prints coverage + harness).
\$RUN python3 make_bad_testset.py \\
    --count ${DATA} \\
    --seed 42 \\
    --out-dir "\$OUTPUT_DIR/bad_testset"

# Run the trained model (+ physics if enabled) over the whole bad testset.
\$RUN python3 src/inference.py \\
    --model-dir "\$OUTPUT_DIR" \\
    --eval-anomaly "\$OUTPUT_DIR/bad_testset/eval_input_anomaly.csv" \\
    --out-dir "\$OUTPUT_DIR/bad_testset_submissions" \\
    ${RF_FLAG} ${PHYS_FLAG}

# Score with the official metric. The run's valid examples are supplied as
# negatives so precision / ROC-AUC are meaningful (not detection recall alone).
\$RUN python3 data/eval_metrics.py \\
    --task anomaly \\
    --ground-truth "\$OUTPUT_DIR/bad_testset/eval_set_forbidden.csv" \\
    --predictions "\$OUTPUT_DIR/bad_testset_submissions/anomaly.csv" \\
    --valid-supplement "\$OUTPUT_DIR/eval_set_valid_supplement.csv"

# Step 5: Plots
echo ""
echo "=== Step 5: Plots ==="
\$RUN python3 src/plot_results.py \\
    --output-dir "\$OUTPUT_DIR"

echo ""
echo "=== DONE — \$(date) ==="
echo "Results in: \$OUTPUT_DIR"
echo "Submissions in: \$OUTPUT_DIR/submissions/"
EOF
