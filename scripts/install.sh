#!/usr/bin/env bash
set -e
module load GCCcore/11.3.0
module load Python/3.10.4
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt