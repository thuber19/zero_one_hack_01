#!/bin/bash
# Run on Leonardo login node: bash setup_and_run.sh
set -e
cd ~/process-sequence-model
rm -rf .pixi pixi.lock
~/.pixi/bin/pixi install
sbatch jobs/train_gpu.sh
