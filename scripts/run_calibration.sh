#!/bin/bash
set -e

DATA_DIR="tracks/industrial-infineon/training_data"
CHECKPOINT="checkpoints/002/checkpoint_best.pt"
SPLITS="checkpoints/002/splits.json"
OUTPUT="checkpoints/002/threshold.json"

echo "==> Generating splits.json ..."
python scripts/generate_splits.py --data-dir "$DATA_DIR" --output "$SPLITS"

echo "==> Calibrating threshold ..."
python scripts/calibrate_threshold.py \
    --checkpoint "$CHECKPOINT" \
    --splits "$SPLITS" \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT" \
    --max-seqs 2000

echo "==> Done: $OUTPUT"
