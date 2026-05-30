#!/bin/bash
# Controlled 3-way training-method comparison (TINY models, few epochs — CPU,
# bounded). Absolute scores are low by design; this isolates which TRAINING
# METHOD wins. The winner gets the full-scale run on Leonardo.
#
#   M0 = baseline            (outputs_test; already trained, no integration)
#   M1 = continue + integration   (fine-tune M0 + UNK-dropout, same vocab)
#   M2 = scratch + integration    (real + pseudo-family data + UNK-dropout)
set -e
cd "$(dirname "$0")"
PY=/c/Users/minam/AppData/Local/Python/bin/python3.14.exe

echo "===== M1: continue from baseline + UNK-dropout ====="
rm -rf outputs_M1; mkdir -p outputs_M1
cp outputs_test/sequences.json outputs_test/tokenizer.txt outputs_M1/
OUTPUT_DIR=outputs_M1 $PY src/train.py --model-size tiny --epochs 5 --device cpu \
  --init-from "$(pwd)/outputs_test/best_transformer.pt" --unk-dropout 0.15

echo "===== M2: from scratch + integrated data (real + pseudo) + UNK-dropout ====="
$PY src/generate_integrated_data.py --extra-data 500 --ood 800 --output-dir outputs_M2
OUTPUT_DIR=outputs_M2 $PY src/train.py --model-size tiny --epochs 6 --device cpu \
  --unk-dropout 0.15

echo "===== COMPARISON TRAINING DONE ====="
