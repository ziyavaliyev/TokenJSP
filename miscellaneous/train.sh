#!/usr/bin/env bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out

export WANDB_API_KEY=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
export PATH="$HOME/.local/bin:$PATH"
uv run python train_gae.py