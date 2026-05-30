#!/bin/bash
# Deploy to Leonardo and submit job
# Run from your Mac: bash tracks/industrial-infineon/deploy.sh

set -e

HOST="a08trd15@login01-ext.leonardo.cineca.it"
REMOTE_DIR="~/process-sequence-model"
LOCAL_DIR="$(dirname "$0")"

echo "=== Syncing files to Leonardo ==="
rsync -av --exclude 'outputs/' --exclude '__pycache__/' --exclude '.pixi/' \
    "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"

echo ""
echo "=== Reinstalling environment and submitting job ==="
ssh "$HOST" "cd $REMOTE_DIR && rm -rf .pixi pixi.lock && ~/.pixi/bin/pixi install && sbatch jobs/train.sh"
