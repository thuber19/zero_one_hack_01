#!/bin/bash
# Second (integrated) training run on Leonardo.
#
# Trains on real + physics-verified pseudo-family data so the model learns
# category-level regularities (Task-4 OOD), then evaluates with the
# physics-integrated inference (refinery + combined validator + fix).
#
# train.py reads its data + writes checkpoints under the OUTPUT_DIR env var and
# trains BOTH the transformer and the Random Forest from sequences.json.
#
# Submit with: sbatch jobs/train_integrated.sh   (tune SBATCH for your quota)
#SBATCH --job-name=proc-integrated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=outputs_integrated/train_%j.log

set -e
cd "$(dirname "$0")/.."

BASE=outputs_integrated

# 1) Build the integrated corpus once (login node; stdlib core, torch optional).
python src/generate_integrated_data.py --extra-data 5000 --ood 1500 --output-dir "$BASE"

# 2) Train at >=2 sizes for the scaling story. Each size gets its own dir with
#    the same data (train.py reads + writes OUTPUT_DIR).
for SIZE in small medium; do
  mkdir -p "$BASE/$SIZE"
  cp "$BASE/sequences.json" "$BASE/tokenizer.txt" "$BASE/$SIZE/" 2>/dev/null || true
  [ -f "$BASE/transitions.json" ] && cp "$BASE/transitions.json" "$BASE/$SIZE/"
  echo "{\"model_size\": \"$SIZE\"}" > "$BASE/$SIZE/model_config.json"
  OUTPUT_DIR="$BASE/$SIZE" python src/train.py --model-size "$SIZE" --epochs 50
done

# 3) Evaluate WITH physics integration (all 3 tasks + per-family); plot curves.
OUTPUT_DIR="$BASE/small" python src/evaluate.py --self-eval --output-dir "$BASE/small"
python src/plot_results.py --output-dir "$BASE/small" \
  --scaling-dirs "$BASE/small" "$BASE/medium" || true

echo "Integrated run complete. Compare vs the first run (outputs/) for baseline-vs-trained."

# RECOMMENDED model-side OOD upgrades (small train.py changes, not yet wired):
#   * UNK-dropout: randomly replace ~10% of step tokens with [UNK] during
#     training so the model predicts from context on unseen tokens — the
#     cleanest 4th-family robustness lever, no vocab bloat.
#   * next-category auxiliary head: a second head predicting classify_step() of
#     the next token (categories from physics/ontology.py), trained jointly.
