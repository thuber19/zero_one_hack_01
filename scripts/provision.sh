#!/usr/bin/env bash
# Note: no 'set -u' because nvm scripts use unbound variables

set -eo pipefail

echo ""
echo "============================================================"
echo " Zero One Hack — Full Login Node Provisioner"
echo "============================================================"
echo ""

# ── 1. Load Python ──────────────────────────────────────────
echo "[1/6] Loading Python 3.11.7..."
module load python/3.11.7
echo "  ✓ $(python3 --version)"

# ── 2. Create venv ──────────────────────────────────────────
echo ""
echo "[2/6] Creating venv at ~/zero_one_env..."
if [[ ! -d "$HOME/zero_one_env" ]]; then
    python3 -m venv "$HOME/zero_one_env"
fi
source "$HOME/zero_one_env/bin/activate"
pip install --upgrade pip --quiet
pip install numpy pandas matplotlib scikit-learn tqdm --quiet
echo "  ✓ venv ready at ~/zero_one_env"
echo "  ✓ numpy, pandas, matplotlib, scikit-learn, tqdm installed"

# ── 3. Install pixi ─────────────────────────────────────────
echo ""
echo "[3/6] Installing pixi..."
if ! command -v pixi &>/dev/null && [[ ! -f "$HOME/.pixi/bin/pixi" ]]; then
    curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$HOME/.pixi/bin:$PATH"
echo "  ✓ pixi $(pixi --version)"

# ── 4. Install Node.js + Claude Code ────────────────────────
echo ""
echo "[4/6] Installing Node.js via nvm..."
if [[ ! -d "$HOME/.nvm" ]]; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi
export NVM_DIR="$HOME/.nvm"
# Load nvm without strict mode (it uses unbound vars internally)
set +u
source "$NVM_DIR/nvm.sh"
nvm install --lts
nvm use --lts
set -u
echo "  ✓ $(node --version)"
echo "  ✓ $(npm --version)"

echo ""
echo "  Installing Claude Code..."
npm install -g @anthropic-ai/claude-code
echo "  ✓ Claude Code installed"

# ── 5. Install starship ─────────────────────────────────────
echo ""
echo "[5/6] Installing starship..."
mkdir -p "$HOME/.local/bin"
curl -sS https://starship.rs/install.sh | sh -s -- --bin-dir "$HOME/.local/bin" --yes
echo "  ✓ $("$HOME/.local/bin/starship" --version)"

echo ""
echo "  Installing antidote..."
if [[ ! -d "$HOME/.antidote" ]]; then
    git clone --depth=1 https://github.com/mattmc3/antidote.git "$HOME/.antidote"
fi
cat > "$HOME/.zsh_plugins.txt" << 'PLUGINS'
zsh-users/zsh-autosuggestions
zsh-users/zsh-syntax-highlighting
zsh-users/zsh-completions
zsh-users/zsh-history-substring-search
PLUGINS
echo "  ✓ antidote + plugins ready"

# ── 6. Write ~/.zshrc ───────────────────────────────────────
echo ""
echo "[6/6] Writing ~/.zshrc..."
cat > "$HOME/.zshrc" << 'ZSHRC'
# PATH
export PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"

# nvm (Node / Claude Code)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

# HPC env
module load python/3.11.7
source ~/zero_one_env/bin/activate

# Antidote plugin manager
source ~/.antidote/antidote.zsh
antidote load ~/.zsh_plugins.txt

# Completion
autoload -Uz compinit
compinit -u
zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}'

# Keybindings
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down
bindkey '^[[C' autosuggest-accept

# Starship prompt
if [[ -f "$HOME/.local/bin/starship" ]]; then
    eval "$($HOME/.local/bin/starship init zsh)"
fi
ZSHRC
echo "  ✓ ~/.zshrc written"

# ── Patch ~/.bashrc to auto-launch zsh ──────────────────────
if ! grep -q "exec zsh" "$HOME/.bashrc" 2>/dev/null; then
    printf '\n# Auto-launch zsh\nexec zsh\n' >> "$HOME/.bashrc"
    echo "  ✓ ~/.bashrc patched to auto-launch zsh"
else
    echo "  ✓ ~/.bashrc already launches zsh"
fi

# ── Done ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " All done! Run:  exec zsh"
echo "============================================================"
echo ""