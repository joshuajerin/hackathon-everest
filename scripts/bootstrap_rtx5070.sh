#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap is for the Linux RTX 5070 workstation." >&2
  exit 1
fi

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found; install a current NVIDIA driver first." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv run python scripts/check_cuda.py
