#!/bin/bash
# Leonardo training + REAL benchmark job (1 GPU). Submit from the project dir:
#   cd $SCRATCH/zero_one_hack_01/tracks/industrial-infineon && sbatch jobs/leonardo/train.slurm
# Scale up: raise --gpus-per-task (<=4) and mem=120GB*gpus, cpus=8*gpus (see PDF).
#SBATCH -J zeroone_train
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

cd "$SCRATCH/zero_one_hack_01/tracks/industrial-infineon"
export PATH="$HOME/.pixi/bin:$PATH"
RUN="$SCRATCH/run_${SLURM_JOB_ID}"
PIXI="pixi run --as-is"
export TRANSFORMERS_OFFLINE=1            # compute nodes are offline; data is in-repo
export PYTHONUNBUFFERED=1                # stream logs live to slurm-<id>.out (no buffering)
NO_RF="--no-rf"                          # skip the slow CPU Random Forest (transformer+physics is the core);
                                         # set NO_RF="" to also train the RF candidate filter.
# If you ever need internet on the compute node (low-bandwidth only — it rotates):
# export HTTPS_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
# export HTTP_PROXY="$HTTPS_PROXY" https_proxy="$HTTPS_PROXY" http_proxy="$HTTPS_PROXY"

MODEL_SIZE=medium
EPOCHS=100

echo "=== 1) data: real (held-out split) + Mina's pseudo-family OOD injection (train-only) ==="
$PIXI python src/generate_data.py --extra-data 5000 --ood 3000 --output-dir "$RUN" --seed 1

echo "=== 2) SFT: aux category head + UNK-dropout (Mina's injections) — model learns patterns + function ==="
OUTPUT_DIR="$RUN" $PIXI python -u src/train.py --arch transformer --model-size "$MODEL_SIZE" \
    --epochs "$EPOCHS" --aux-category --unk-dropout 0.15 $NO_RF --device cuda

echo "=== 3) GRPO: push the constraints into the weights (verifier reward) ==="
OUTPUT_DIR="$RUN" $PIXI python src/train_grpo.py --init-from "$RUN/best_model.pt" \
    --data-dir "$RUN" --steps 3000 --group-size 8 --device cuda || echo "(GRPO skipped/failed — SFT model still usable)"

echo "=== 4) inference on the HELD-OUT eval -> submissions ==="
$PIXI python src/inference.py --output-dir "$RUN" --eval-dir "$RUN" --model-size "$MODEL_SIZE"

echo "=== 5) OFFICIAL scoring (real T1/T2/T3 numbers) ==="
$PIXI python data/eval_metrics.py --task next-step  --ground-truth "$RUN/eval_set_valid.csv"     --predictions "$RUN/submissions/nextstep.csv"   || true
$PIXI python data/eval_metrics.py --task completion --ground-truth "$RUN/eval_set_valid.csv"     --predictions "$RUN/submissions/completion.csv" || true
$PIXI python data/eval_metrics.py --task anomaly    --ground-truth "$RUN/eval_set_forbidden.csv" --predictions "$RUN/submissions/anomaly.csv" \
       --valid-supplement "$RUN/eval_set_valid_supplement.csv" || true

echo "=== 6) (optional) model functional understanding ==="
OUTPUT_DIR="$RUN" $PIXI python category_eval.py --model-dir "$RUN" --model-size "$MODEL_SIZE" || true

echo "DONE. Checkpoints + submissions in: $RUN"
echo "For the ACTUAL submission, run inference on the organizers' files:"
echo "  $PIXI python src/inference.py --output-dir \"$RUN\" --eval-dir data --model-size $MODEL_SIZE"
