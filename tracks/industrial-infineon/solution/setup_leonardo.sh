#!/bin/bash
# Run ONCE on the Leonardo LOGIN node (it has internet) to build the procseq venv.
# Usage (from tracks/industrial-infineon/solution/):
#   bash setup_leonardo.sh
#
# No DeepSpeed / no `module load cuda` needed: the runs are single-GPU + plain
# Accelerate, and the pip torch wheel ships its own CUDA runtime for the A100s.
set -euo pipefail

VENV="${VENV:-$HOME/procseq-venv}"

# Need a Python >= 3.10. CINECA provides it via modules; the exact name varies —
# if the first load fails, run `module avail python` and load the right one, or
# reuse pixi's python ($HOME/.pixi/bin/python). pip itself needs login-node net.
module load python/3.11.6--gcc--12.2.0 2>/dev/null \
  || module load python/3.11 2>/dev/null \
  || module load python 2>/dev/null \
  || echo "[warn] no python module loaded; falling back to system python3"
PY="$(command -v python3 || command -v python)"
echo "Using python: $PY ($($PY --version 2>&1))"

"$PY" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  "torch>=2.4" "transformers>=4.44" "tokenizers>=0.19" "accelerate>=0.33" \
  "numpy>=1.26" "matplotlib>=3.8" "pyyaml>=6.0" "tensorboard"

python -c "import torch, transformers, accelerate, tokenizers; \
print('ok:', 'torch', torch.__version__, '| transformers', transformers.__version__)"

echo ""
echo "venv ready at: $VENV"
echo "Next (from this solution dir):"
echo "  sbatch slurm/train_decoder.sbatch"
echo "  sbatch slurm/train_encoder.sbatch"
