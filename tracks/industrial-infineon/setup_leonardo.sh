#!/bin/bash
# Run this on the Leonardo LOGIN NODE (not compute node)
# It sets up the pixi environment and generates training data.

set -e

echo "=== Setting up project on Leonardo ==="

# Install pixi if not present
if ! command -v pixi &> /dev/null; then
    echo "Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

# Navigate to project directory
PROJ_DIR="$HOME/process-sequence-model"
mkdir -p "$PROJ_DIR"

echo "Project directory: $PROJ_DIR"
echo "Copy your project files here, then run:"
echo "  cd $PROJ_DIR"
echo "  pixi install"
echo ""
echo "Then generate training data (can run on login node, no GPU needed):"
echo "  pixi run python src/generate_data.py"
echo ""
echo "Then submit the training job:"
echo "  sbatch jobs/train.sh"
