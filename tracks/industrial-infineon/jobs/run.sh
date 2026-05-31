#!/bin/bash
#
# Job launcher for Leonardo.
# Usage: bash jobs/run.sh
#

PROJECT_DIR="$HOME/process-sequence-model"
PIXI="$HOME/.pixi/bin/pixi"
DATASETS_DIR="${SCRATCH:-$HOME}/datasets"
RUNS_DIR="${SCRATCH:-$HOME}/runs"
SOL="$PROJECT_DIR/solution"

echo "============================================"
echo "  Process Sequence Model — Job Launcher"
echo "============================================"
echo ""
echo "  1) Generate dataset (train + eval split)"
echo "  2) Train LSTM/Transformer"
echo "  3) Train procseq (decoder + encoder, all 3 tasks)"
echo "  4) Run inference on any checkpoint"
echo "  5) Score submission files"
echo ""
read -p "Choice [1-5]: " CHOICE

# =========================================================================
# 1) Generate dataset
# =========================================================================
if [ "$CHOICE" = "1" ]; then
    read -p "Sequences per family [20000]: " DATA; DATA="${DATA:-20000}"
    read -p "Eval split [0.05]: " SPLIT; SPLIT="${SPLIT:-0.05}"
    read -p "Dataset name [default]: " DSNAME; DSNAME="${DSNAME:-default}"
    DS_DIR="$DATASETS_DIR/$DSNAME"

    echo "Generating to: $DS_DIR"
    mkdir -p "$DS_DIR"
    cd "$PROJECT_DIR"
    "$PIXI" run --manifest-path pixi.toml python3 src/generate_data.py \
        --extra-data "$DATA" --output-dir "$DS_DIR" --eval-split "$SPLIT" --seed 42
    echo "Done: $DS_DIR"
    exit 0
fi

# =========================================================================
# 2) Train LSTM/Transformer
# =========================================================================
if [ "$CHOICE" = "2" ]; then
    echo "  1) transformer-tiny  2) transformer-small  3) transformer-medium"
    echo "  4) lstm-tiny         5) lstm-small         6) lstm-medium"
    read -p "Choice [1-6]: " MC
    case $MC in
        1) ARCH="transformer"; SIZE="tiny" ;; 2) ARCH="transformer"; SIZE="small" ;;
        3) ARCH="transformer"; SIZE="medium" ;; 4) ARCH="lstm"; SIZE="tiny" ;;
        5) ARCH="lstm"; SIZE="small" ;; 6) ARCH="lstm"; SIZE="medium" ;;
        *) echo "Invalid"; exit 1 ;;
    esac
    read -p "Dataset name [default]: " DSNAME; DSNAME="${DSNAME:-default}"
    read -p "Epochs [100]: " EPOCHS; EPOCHS="${EPOCHS:-100}"
    read -p "Batch size [512]: " BATCH; BATCH="${BATCH:-512}"
    echo "Random Forest? 1=Yes 2=No [1]: "; read RC; RC="${RC:-1}"
    [ "$RC" = "2" ] && RF="--no-rf" || RF=""

    DS_DIR="$DATASETS_DIR/$DSNAME"
    [ ! -f "$DS_DIR/train_sequences.csv" ] && echo "ERROR: Dataset not found. Run option 1." && exit 1
    RUN_NAME="${ARCH}_${SIZE}_e${EPOCHS}_${DSNAME}"
    RUN_DIR="$RUNS_DIR/$RUN_NAME"

    echo "  Model: $ARCH $SIZE | Dataset: $DS_DIR | Output: $RUN_DIR"
    read -p "Submit? [Y/n]: " C; C="${C:-Y}"
    [[ "$C" == "Y" || "$C" == "y" ]] || exit 0

    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-task=1
#SBATCH --mem=120GB --cpus-per-task=8 --time=3:00:00
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=slurm-${RUN_NAME}-%j.out
#SBATCH --error=slurm-${RUN_NAME}-%j.err
set -e
export PYTHONUNBUFFERED=1 OUTPUT_DIR="$RUN_DIR"
mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"
RUN="$PIXI run --as-is --manifest-path $PROJECT_DIR/pixi.toml env PYTHONUNBUFFERED=1"
cp "$DS_DIR/tokenizer.txt" "$RUN_DIR/"
cp "$DS_DIR/train_sequences.csv" "$RUN_DIR/"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
\$RUN python3 src/train.py --arch $ARCH --model-size $SIZE --epochs $EPOCHS --batch-size $BATCH --lr 3e-4 --seed 42 $RF
\$RUN python3 src/inference.py --model-dir "$RUN_DIR" --eval-valid "$DS_DIR/eval_input_valid.csv" --eval-anomaly "$DS_DIR/eval_input_anomaly.csv" --out-dir "$RUN_DIR/self_eval"
\$RUN python3 src/inference.py --model-dir "$RUN_DIR" --eval-valid "$PROJECT_DIR/data/eval_input_valid.csv" --eval-anomaly "$PROJECT_DIR/data/eval_input_anomaly.csv" --out-dir "$RUN_DIR/official"
\$RUN python3 data/eval_metrics.py --task next-step --ground-truth "$DS_DIR/eval_set_valid.csv" --predictions "$RUN_DIR/self_eval/nextstep.csv"
\$RUN python3 data/eval_metrics.py --task completion --ground-truth "$DS_DIR/eval_set_valid.csv" --predictions "$RUN_DIR/self_eval/completion.csv"
\$RUN python3 data/eval_metrics.py --task anomaly --ground-truth "$DS_DIR/eval_set_forbidden.csv" --predictions "$RUN_DIR/self_eval/anomaly.csv" --valid-supplement "$DS_DIR/eval_set_valid_supplement.csv"
echo "=== DONE ==="
EOF
    exit 0
fi

# =========================================================================
# 3) Train procseq (full: decoder + encoder)
# =========================================================================
if [ "$CHOICE" = "3" ]; then
    read -p "Model size [base]: " PSIZE; PSIZE="${PSIZE:-base}"
    read -p "Sequences per family [20000]: " PDATA; PDATA="${PDATA:-20000}"
    read -p "Max steps [8000]: " PSTEPS; PSTEPS="${PSTEPS:-8000}"
    read -p "Batch size [64]: " PBATCH; PBATCH="${PBATCH:-64}"

    [ ! -d "$SOL" ] && echo "ERROR: $SOL not found." && exit 1

    SEED=$RANDOM
    OUTNAME="procseq_${PSIZE}_d${PDATA}_s${PSTEPS}_seed${SEED}"
    OUTDIR="${SCRATCH:-$HOME}/runs_no_seed/$OUTNAME"
    CFG="$SOL/configs/_run_${OUTNAME}.yaml"

    echo "Installing procseq pixi env..."
    "$PIXI" install --manifest-path "$PROJECT_DIR/pixi.toml" -e procseq

    mkdir -p "$OUTDIR"
    cat > "$CFG" <<YAML
run_name: ${OUTNAME}
seed: ${SEED}
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

    echo "  procseq $PSIZE | ${PDATA}/family | ${PSTEPS} steps | output: $OUTDIR"
    read -p "Submit? [Y/n]: " C; C="${C:-Y}"
    [[ "$C" == "Y" || "$C" == "y" ]] || exit 0

    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-task=1
#SBATCH --mem=120GB --cpus-per-task=8 --time=3:00:00
#SBATCH --job-name=procseq-${PSIZE}
#SBATCH --output=slurm-${OUTNAME}-%j.out
#SBATCH --error=slurm-${OUTNAME}-%j.err
set -e
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
RUN="${PIXI} run --as-is --manifest-path ${PROJECT_DIR}/pixi.toml -e procseq env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=${SOL} PROCSEQ_ARTIFACTS=${OUTDIR}"
cd "${SOL}"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
\$RUN python3 -m procseq.run_all --config "${CFG}"
echo "=== DONE ==="
echo "Submissions: ${OUTDIR}/submission_task*_real.csv"
EOF
    exit 0
fi

# =========================================================================
# 4) Run inference on any checkpoint
# =========================================================================
if [ "$CHOICE" = "4" ]; then
    echo ""
    echo "Available runs:"
    ls -1d "$RUNS_DIR"/*/ 2>/dev/null | while read d; do basename "$d"; done
    echo ""
    read -p "Run name: " RUN_NAME
    RUN_DIR="$RUNS_DIR/$RUN_NAME"

    # Detect which pipeline
    if [ -d "$RUN_DIR/decoder" ]; then
        TYPE="procseq"
    elif [ -f "$RUN_DIR/best_model.pt" ]; then
        TYPE="ours"
    else
        echo "ERROR: No checkpoint found in $RUN_DIR"
        exit 1
    fi

    echo "Detected: $TYPE model"
    echo "Eval on: 1) Self-eval  2) Official test set"
    read -p "Choice [1-2]: " EC

    echo "Installing pixi env..."
    if [ "$TYPE" = "procseq" ]; then
        "$PIXI" install --manifest-path "$PROJECT_DIR/pixi.toml" -e procseq
    fi

    read -p "Submit? [Y/n]: " C; C="${C:-Y}"
    [[ "$C" == "Y" || "$C" == "y" ]] || exit 0

    if [ "$EC" = "1" ]; then
        read -p "Dataset name [default]: " DSNAME; DSNAME="${DSNAME:-default}"
        DS_DIR="$DATASETS_DIR/$DSNAME"
        EVAL_V="$DS_DIR/eval_input_valid.csv"
        EVAL_A="$DS_DIR/eval_input_anomaly.csv"
        OUTDIR="$RUN_DIR/self_eval"
    else
        EVAL_V="$PROJECT_DIR/data/eval_input_valid.csv"
        EVAL_A="$PROJECT_DIR/data/eval_input_anomaly.csv"
        OUTDIR="$RUN_DIR/official"
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-task=1
#SBATCH --mem=120GB --cpus-per-task=8 --time=1:00:00
#SBATCH --job-name=eval-${RUN_NAME}
#SBATCH --output=slurm-eval-${RUN_NAME}-%j.out
#SBATCH --error=slurm-eval-${RUN_NAME}-%j.err
set -e
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"

if [ "$TYPE" = "procseq" ]; then
    # Write eval config pointing to this checkpoint
    cat > "$SOL/configs/_run_eval.yaml" <<YCFG
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
YCFG
    export TOKENIZERS_PARALLELISM=false
    RUN="${PIXI} run --as-is --manifest-path ${PROJECT_DIR}/pixi.toml -e procseq env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=${SOL} PROCSEQ_ARTIFACTS=${RUN_DIR}"
    cd "${SOL}"
    \$RUN python3 -m procseq.infer --task 1 --config "${SOL}/configs/_run_eval.yaml"
    \$RUN python3 -m procseq.infer --task 2 --config "${SOL}/configs/_run_eval.yaml"
    \$RUN python3 -m procseq.run_eval --config "${SOL}/configs/_run_eval.yaml"
    if [ "$EC" = "2" ]; then
        \$RUN python3 -m procseq.infer --task 1 --real --config "${SOL}/configs/_run_eval.yaml"
        \$RUN python3 -m procseq.infer --task 2 --real --config "${SOL}/configs/_run_eval.yaml"
    fi
else
    RUN="$PIXI run --as-is --manifest-path $PROJECT_DIR/pixi.toml env PYTHONUNBUFFERED=1"
    \$RUN python3 src/inference.py --model-dir "$RUN_DIR" --eval-valid "$EVAL_V" --eval-anomaly "$EVAL_A" --out-dir "$OUTDIR"
fi
echo "=== DONE ==="
EOF
    exit 0
fi

# =========================================================================
# 5) Score submission files
# =========================================================================
if [ "$CHOICE" = "5" ]; then
    read -p "Path to submission dir: " SUB_DIR
    [ ! -d "$SUB_DIR" ] && echo "ERROR: not found" && exit 1
    read -p "Dataset name [default]: " DSNAME; DSNAME="${DSNAME:-default}"
    DS_DIR="$DATASETS_DIR/$DSNAME"
    [ ! -f "$DS_DIR/eval_set_valid.csv" ] && echo "ERROR: No ground truth at $DS_DIR" && exit 1

    # Auto-detect file names
    NS="$SUB_DIR/nextstep.csv"; [ ! -f "$NS" ] && NS="$SUB_DIR/submission_task1.csv"
    CO="$SUB_DIR/completion.csv"; [ ! -f "$CO" ] && CO="$SUB_DIR/submission_task2.csv"
    AN="$SUB_DIR/anomaly.csv"; [ ! -f "$AN" ] && AN="$SUB_DIR/submission_task3.csv"

    cd "$PROJECT_DIR"
    [ -f "$NS" ] && echo "--- Task 1 ---" && "$PIXI" run --manifest-path pixi.toml python3 data/eval_metrics.py --task next-step --ground-truth "$DS_DIR/eval_set_valid.csv" --predictions "$NS"
    [ -f "$CO" ] && echo "--- Task 2 ---" && "$PIXI" run --manifest-path pixi.toml python3 data/eval_metrics.py --task completion --ground-truth "$DS_DIR/eval_set_valid.csv" --predictions "$CO"
    [ -f "$AN" ] && echo "--- Task 3 ---" && "$PIXI" run --manifest-path pixi.toml python3 data/eval_metrics.py --task anomaly --ground-truth "$DS_DIR/eval_set_forbidden.csv" --predictions "$AN" --valid-supplement "$DS_DIR/eval_set_valid_supplement.csv"
    exit 0
fi

echo "Invalid choice."
exit 1
