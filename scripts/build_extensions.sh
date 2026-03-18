#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -c "import torch; print('torch:', torch.__version__)"

echo "[Dive3D] Building CUDA extensions (raymarching / gridencoder / freqencoder)"
pip install -U pip setuptools wheel ninja

pip install -e ./raymarching
pip install -e ./gridencoder
pip install -e ./freqencoder

echo "[Dive3D] Done."
