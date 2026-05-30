#!/bin/bash
# One-time setup on a Leonardo LOGIN node (which HAS internet).
# Hackathon storage rule (PDF): use $SCRATCH for big files; NOT $WORK / $FAST.
# Login nodes kill processes after 10 min CPU — if pixi install is slow, run this
# under: srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash
set -euo pipefail

REPO_URL="https://github.com/thuber19/zero_one_hack_01.git"
BRANCH="mina/physics-integration"
DEST="${SCRATCH:?SCRATCH not set — are you on Leonardo?}/zero_one_hack_01"

echo "[1/4] clone/update repo into \$SCRATCH"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch origin && git -C "$DEST" checkout "$BRANCH" && git -C "$DEST" pull --ff-only
else
  git clone "$REPO_URL" "$DEST" && git -C "$DEST" checkout "$BRANCH"
fi

echo "[2/4] move caches off the 50GB \$HOME quota onto \$SCRATCH"
mkdir -p "$SCRATCH/.cache"
if [ ! -L "$HOME/.cache" ]; then rm -rf "$HOME/.cache"; ln -s "$SCRATCH/.cache" "$HOME/.cache"; fi
export PIXI_CACHE_DIR="$SCRATCH/.cache/pixi"

echo "[3/4] install pixi (per hackathon doc) if missing"
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi

echo "[4/4] build the project env (python 3.11 + torch/cuda12 + numpy + sklearn)"
cd "$DEST/tracks/industrial-infineon"
[ -f pixi.toml ] || pixi init --quiet
pixi workspace system-requirements add cuda 12 >/dev/null 2>&1 || true
pixi add python=3.11 >/dev/null 2>&1 || true
CONDA_OVERRIDE_CUDA=12 pixi add --pypi torch numpy scikit-learn >/dev/null 2>&1 || \
  pixi add --pypi torch numpy scikit-learn || true

echo "DONE. Project at: $DEST/tracks/industrial-infineon"
echo "Next:  cd $DEST/tracks/industrial-infineon && sbatch jobs/leonardo/train.slurm"
