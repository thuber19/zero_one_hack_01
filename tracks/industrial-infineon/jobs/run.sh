#!/bin/bash
#
# Job launcher for Leonardo.
#
# Data is generated ONCE into $SCRATCH/datasets/<name>/ and reused.
# Models are trained into $SCRATCH/runs/<name>/ with checkpoints.
# Eval can be re-run on any checkpoint without retraining.
#
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

    echo ""
    echo "Dataset ready at: $DS_DIR"
    echo "Files: train_sequences.csv, eval_input_valid.csv, eval_input_anomaly.csv,"
    echo "       eval_set_valid.csv, eval_set_forbidden.csv, tokenizer.txt"
    exit 0
fi

# =========================================================================
# 2) Train LSTM/Transformer
# =========================================================================
if [ "$CHOICE" = "2" ]; then
    echo ""
    echo "Select model:"
    echo "  1) transformer-tiny    2) transformer-small   3) transformer-medium"
    echo "  4) lstm-tiny           5) lstm-small          6) lstm-medium"
    read -p "Choice [1-6]: " MC
    case $MC in
        1) ARCH="transformer"; SIZE="tiny" ;;
        2) ARCH="transformer"; SIZE="small" ;;
        3) ARCH="transformer"; SIZE="medium" ;;
        4) ARCH="lstm"; SIZE="tiny" ;;
        5) ARCH="lstm"; SIZE="small" ;;
        6) ARCH="lstm"; SIZE="medium" ;;
        *) echo "Invalid"; exit 1 ;;
    esac

    read -p "Dataset name [default]: " DSNAME; DSNAME="${DSNAME:-default}"
    read -p "Epochs [100]: " EPOCHS; EPOCHS="${EPOCHS:-100}"
    read -p "Batch size [512]: " BATCH; BATCH="${BATCH:-512}"
    echo "Train Random Forest? 1=Yes 2=No [1]: "; read RF_C; RF_C="${RF_C:-1}"
    [ "$RF_C" = "2" ] && RF_FLAG="--no-rf" || RF_FLAG=""

    DS_DIR="$DATASETS_DIR/$DSNAME"
    if [ ! -f "$DS_DIR/train_sequences.csv" ]; then
        echo "ERROR: Dataset not found at $DS_DIR. Run option 1 first."
        exit 1
    fi

    RUN_NAME="${ARCH}_${SIZE}_e${EPOCHS}_$(basename $DS_DIR)"
    RUN_DIR="$RUNS_DIR/$RUN_NAME"

    echo ""
    echo "  Model:   $ARCH $SIZE ($EPOCHS epochs, batch $BATCH)"
    echo "  Dataset: $DS_DIR"
    echo "  Output:  $RUN_DIR"
    read -p "Submit? [Y/n]: " C; C="${C:-Y}"
    [[ "$C" == "Y" || "$C" == "y" ]] || exit 0

    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=slurm-${RUN_NAME}-%j.out
#SBATCH --error=slurm-${RUN_NAME}-%j.err

set -e
export PYTHONUNBUFFERED=1
export OUTPUT_DIR="$RUN_DIR"
mkdir -p "\$OUTPUT_DIR"
cd "$PROJECT_DIR"

RUN="$PIXI run --as-is --manifest-path $PROJECT_DIR/pixi.toml env PYTHONUNBUFFERED=1"

echo "=== Training $ARCH $SIZE ==="
echo "Dataset: $DS_DIR"
echo "Output:  \$OUTPUT_DIR"
echo "GPU:     \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo ""

# Copy dataset files to run dir so train.py can find them
cp "$DS_DIR/tokenizer.txt" "\$OUTPUT_DIR/"
cp "$DS_DIR/train_sequences.csv" "\$OUTPUT_DIR/"

\$RUN python3 src/train.py \\
    --arch $ARCH \\
    --model-size $SIZE \\
    --epochs $EPOCHS \\
    --batch-size $BATCH \\
    --lr 3e-4 \\
    --seed 42 $RF_FLAG

echo ""
echo "=== Training done. Running self-eval... ==="

\$RUN python3 src/inference.py \\
    --model-dir "\$OUTPUT_DIR" \\
    --eval-valid "$DS_DIR/eval_input_valid.csv" \\
    --eval-anomaly "$DS_DIR/eval_input_anomaly.csv" \\
    --out-dir "\$OUTPUT_DIR/self_eval"

\$RUN python3 data/eval_metrics.py \\
    --task next-step \\
    --ground-truth "$DS_DIR/eval_set_valid.csv" \\
    --predictions "\$OUTPUT_DIR/self_eval/nextstep.csv"

\$RUN python3 data/eval_metrics.py \\
    --task completion \\
    --ground-truth "$DS_DIR/eval_set_valid.csv" \\
    --predictions "\$OUTPUT_DIR/self_eval/completion.csv"

\$RUN python3 data/eval_metrics.py \\
    --task anomaly \\
    --ground-truth "$DS_DIR/eval_set_forbidden.csv" \\
    --predictions "\$OUTPUT_DIR/self_eval/anomaly.csv" \\
    --valid-supplement "$DS_DIR/eval_set_valid_supplement.csv"

echo ""
echo "=== Running official eval... ==="

\$RUN python3 src/inference.py \\
    --model-dir "\$OUTPUT_DIR" \\
    --eval-valid "$PROJECT_DIR/data/eval_input_valid.csv" \\
    --eval-anomaly "$PROJECT_DIR/data/eval_input_anomaly.csv" \\
    --out-dir "\$OUTPUT_DIR/official_submissions"

echo ""
echo "=== DONE ==="
echo "Self-eval:    \$OUTPUT_DIR/self_eval/"
echo "Official sub: \$OUTPUT_DIR/official_submissions/"
EOF
    echo "Submitted. Watch: tail -f slurm-${RUN_NAME}-*.out"
    exit 0
fi

# =========================================================================
# 3) Train procseq decoder
# =========================================================================
if [ "$CHOICE" = "3" ]; then
    read -p "Model size [base]: " PSIZE; PSIZE="${PSIZE:-base}"
    read -p "Sequences per family [20000]: " PDATA; PDATA="${PDATA:-20000}"
    read -p "Max training steps [8000]: " PSTEPS; PSTEPS="${PSTEPS:-8000}"
    read -p "Batch size [64]: " PBATCH; PBATCH="${PBATCH:-64}"

    SOL="$PROJECT_DIR/solution"
    OUTNAME="procseq_decoder_${PSIZE}_d${PDATA}_s${PSTEPS}"
    OUTDIR="$RUNS_DIR/$OUTNAME"
    CFG="$SOL/configs/_run_decoder.yaml"

    [ ! -d "$SOL" ] && echo "ERROR: $SOL not found." && exit 1

    echo "Installing procseq pixi env..."
    "$PIXI" install --manifest-path "$PROJECT_DIR/pixi.toml" -e procseq

    mkdir -p "$OUTDIR"
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
    [[ "$C" == "Y" || "$C" == "y" ]] || exit 0

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
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
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

# =========================================================================
# 4) Train procseq encoder
# =========================================================================
if [ "$CHOICE" = "4" ]; then
    read -p "Model size [base]: " PSIZE; PSIZE="${PSIZE:-base}"
    read -p "Sequences per family [20000]: " PDATA; PDATA="${PDATA:-20000}"
    read -p "Max training steps [4000]: " PSTEPS; PSTEPS="${PSTEPS:-4000}"
    read -p "Batch size [64]: " PBATCH; PBATCH="${PBATCH:-64}"

    SOL="$PROJECT_DIR/solution"
    OUTNAME="procseq_encoder_${PSIZE}_d${PDATA}_s${PSTEPS}"
    OUTDIR="$RUNS_DIR/$OUTNAME"
    CFG="$SOL/configs/_run_encoder.yaml"

    [ ! -d "$SOL" ] && echo "ERROR: $SOL not found." && exit 1

    "$PIXI" install --manifest-path "$PROJECT_DIR/pixi.toml" -e procseq

    mkdir -p "$OUTDIR"
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
  lr: 0.0005
  max_steps: ${PSTEPS}
  contrastive:
    enabled: true
    weight: 0.5
    temperature: 0.1
YAML

    echo ""
    echo "  procseq encoder | $PSIZE | ${PDATA}/family | ${PSTEPS} steps"
    echo "  output: $OUTDIR"
    read -p "Submit? [Y/n]: " C; C="${C:-Y}"
    [[ "$C" == "Y" || "$C" == "y" ]] || exit 0

    sbatch <<EOF
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
#SBATCH --job-name=procseq-enc
#SBATCH --output=slurm-${OUTNAME}-%j.out
#SBATCH --error=slurm-${OUTNAME}-%j.err

set -e
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
RUN="${PIXI} run --as-is --manifest-path ${PROJECT_DIR}/pixi.toml -e procseq env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=${SOL} PROCSEQ_ARTIFACTS=${OUTDIR}"
cd "${SOL}"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

\$RUN python3 -m procseq.build_data --n-per-family ${PDATA} --seed 42
\$RUN accelerate launch --num_processes 1 --mixed_precision bf16 -m procseq.train_encoder --config "${CFG}"
\$RUN python3 -m procseq.infer --task 3 --config "${CFG}"
\$RUN python3 -m procseq.run_eval --config "${CFG}"
\$RUN python3 -m procseq.infer --task 3 --real --config "${CFG}"

echo "=== DONE ==="
echo "Results: ${OUTDIR}/metrics.json"
echo "Submissions: ${OUTDIR}/submission_task3_real.csv"
EOF
    echo "Submitted. Watch: tail -f slurm-${OUTNAME}-*.out"
    exit 0
fi

# =========================================================================
# 5) Run inference on our LSTM/Transformer checkpoint
# =========================================================================
if [ "$CHOICE" = "5" ]; then
    echo ""
    echo "Available runs:"
    ls -d "$RUNS_DIR"/lstm_* "$RUNS_DIR"/transformer_* 2>/dev/null | while read d; do basename "$d"; done
    echo ""
    read -p "Run name: " RUN_NAME
    RUN_DIR="$RUNS_DIR/$RUN_NAME"
    [ ! -f "$RUN_DIR/best_model.pt" ] && echo "ERROR: No checkpoint at $RUN_DIR/best_model.pt" && exit 1

    echo ""
    echo "Eval on:"
    echo "  1) Self-eval (from dataset)"
    echo "  2) Official eval (organizer files)"
    read -p "Choice [1-2]: " EC

    echo "Use physics? 1=No 2=Yes [1]: "; read PC; PC="${PC:-1}"
    [ "$PC" = "2" ] && PHYS="--physics" || PHYS=""

    if [ "$EC" = "1" ]; then
        read -p "Dataset name [default]: " DSNAME; DSNAME="${DSNAME:-default}"
        DS_DIR="$DATASETS_DIR/$DSNAME"
        EVAL_VALID="$DS_DIR/eval_input_valid.csv"
        EVAL_ANOMALY="$DS_DIR/eval_input_anomaly.csv"
        OUT_DIR="$RUN_DIR/self_eval"
    else
        EVAL_VALID="$PROJECT_DIR/data/eval_input_valid.csv"
        EVAL_ANOMALY="$PROJECT_DIR/data/eval_input_anomaly.csv"
        OUT_DIR="$RUN_DIR/official_submissions"
    fi

    echo ""
    echo "  Checkpoint: $RUN_DIR"
    echo "  Eval files: $EVAL_VALID"
    echo "  Output:     $OUT_DIR"
    read -p "Submit? [Y/n]: " C; C="${C:-Y}"
    [[ "$C" == "Y" || "$C" == "y" ]] || exit 0

    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=eval-${RUN_NAME}
#SBATCH --output=slurm-eval-${RUN_NAME}-%j.out
#SBATCH --error=slurm-eval-${RUN_NAME}-%j.err

set -e
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"
RUN="$PIXI run --as-is --manifest-path $PROJECT_DIR/pixi.toml env PYTHONUNBUFFERED=1"

\$RUN python3 src/inference.py \\
    --model-dir "$RUN_DIR" \\
    --eval-valid "$EVAL_VALID" \\
    --eval-anomaly "$EVAL_ANOMALY" \\
    --out-dir "$OUT_DIR" \\
    $PHYS

echo "=== Inference done. Submissions in: $OUT_DIR ==="
EOF
    echo "Submitted."
    exit 0
fi

# =========================================================================
# 6) Run inference on procseq checkpoint
# =========================================================================
if [ "$CHOICE" = "6" ]; then
    echo ""
    echo "Available procseq runs:"
    ls -d "$RUNS_DIR"/procseq_decoder_* 2>/dev/null | while read d; do basename "$d"; done
    echo ""
    read -p "Run name: " RUN_NAME
    RUN_DIR="$RUNS_DIR/$RUN_NAME"
    [ ! -d "$RUN_DIR/decoder" ] && echo "ERROR: No decoder checkpoint at $RUN_DIR/decoder" && exit 1

    SOL="$PROJECT_DIR/solution"
    CFG="$SOL/configs/_run_eval.yaml"

    # Write config pointing to the selected checkpoint
    cat > "$CFG" <<EVALYAML
run_name: eval_${RUN_NAME}
seed: 42
precision: bf16
artifacts: ${RUN_DIR}
decoder_ckpt: ${RUN_DIR}/decoder
encoder_ckpt: ${RUN_DIR}/encoder
decoder:
  size: base
  max_len: 256
  constrained_decode: true
EVALYAML

    echo "Ensuring procseq pixi env is installed..."
    "$PIXI" install --manifest-path "$PROJECT_DIR/pixi.toml" -e procseq

    echo ""
    echo "  Checkpoint: $RUN_DIR/decoder"
    echo "  Config:     $CFG (auto-generated)"
    read -p "Submit? [Y/n]: " C; C="${C:-Y}"
    [[ "$C" == "Y" || "$C" == "y" ]] || exit 0

    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=eval-procseq
#SBATCH --output=slurm-eval-procseq-%j.out
#SBATCH --error=slurm-eval-procseq-%j.err

set -e
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
RUN="${PIXI} run --as-is --manifest-path ${PROJECT_DIR}/pixi.toml -e procseq env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=${SOL} PROCSEQ_ARTIFACTS=${RUN_DIR}"
cd "${SOL}"

\$RUN python3 -m procseq.infer --task 1 --config "${CFG}"
\$RUN python3 -m procseq.infer --task 2 --config "${CFG}"
\$RUN python3 -m procseq.run_eval --config "${CFG}"
\$RUN python3 -m procseq.infer --task 1 --real --config "${CFG}"
\$RUN python3 -m procseq.infer --task 2 --real --config "${CFG}"

echo "=== DONE ==="
echo "Results: ${RUN_DIR}/metrics.json"
echo "Submissions: ${RUN_DIR}/submission_task*_real.csv"
EOF
    echo "Submitted."
    exit 0
fi

# =========================================================================
# 7) Score any submission files with eval_metrics.py
# =========================================================================
if [ "$CHOICE" = "7" ]; then
    echo ""
    echo "Score submission CSVs against ground truth."
    echo ""
    read -p "Path to submission dir (contains nextstep.csv/completion.csv/anomaly.csv): " SUB_DIR
    [ ! -d "$SUB_DIR" ] && echo "ERROR: $SUB_DIR not found" && exit 1

    # Auto-detect submission file names (ours vs procseq naming)
    NEXTSTEP="$SUB_DIR/nextstep.csv"
    [ ! -f "$NEXTSTEP" ] && NEXTSTEP="$SUB_DIR/submission_task1.csv"
    COMPLETION="$SUB_DIR/completion.csv"
    [ ! -f "$COMPLETION" ] && COMPLETION="$SUB_DIR/submission_task2.csv"
    ANOMALY="$SUB_DIR/anomaly.csv"
    [ ! -f "$ANOMALY" ] && ANOMALY="$SUB_DIR/submission_task3.csv"

    echo ""
    echo "Which ground truth?"
    echo "  1) Self-eval (from dataset)"
    echo "  2) None (just check format)"
    read -p "Choice [1-2]: " GTC
    GTC="${GTC:-1}"

    if [ "$GTC" = "1" ]; then
        read -p "Dataset name [default]: " DSNAME; DSNAME="${DSNAME:-default}"
        DS_DIR="$DATASETS_DIR/$DSNAME"
        [ ! -f "$DS_DIR/eval_set_valid.csv" ] && echo "ERROR: No ground truth at $DS_DIR" && exit 1

        cd "$PROJECT_DIR"

        echo ""
        if [ -f "$NEXTSTEP" ]; then
            echo "--- Task 1: Next-step ---"
            "$PIXI" run --manifest-path pixi.toml python3 data/eval_metrics.py \
                --task next-step \
                --ground-truth "$DS_DIR/eval_set_valid.csv" \
                --predictions "$NEXTSTEP"
        fi

        if [ -f "$COMPLETION" ]; then
            echo ""
            echo "--- Task 2: Completion ---"
            "$PIXI" run --manifest-path pixi.toml python3 data/eval_metrics.py \
                --task completion \
                --ground-truth "$DS_DIR/eval_set_valid.csv" \
                --predictions "$COMPLETION"
        fi

        if [ -f "$ANOMALY" ]; then
            echo ""
            echo "--- Task 3: Anomaly ---"
            "$PIXI" run --manifest-path pixi.toml python3 data/eval_metrics.py \
                --task anomaly \
                --ground-truth "$DS_DIR/eval_set_forbidden.csv" \
                --predictions "$ANOMALY" \
                --valid-supplement "$DS_DIR/eval_set_valid_supplement.csv"
        fi
    else
        echo "No ground truth — just listing files:"
        ls -la "$SUB_DIR"/*.csv 2>/dev/null
    fi

    exit 0
fi

echo "Invalid choice."
exit 1
