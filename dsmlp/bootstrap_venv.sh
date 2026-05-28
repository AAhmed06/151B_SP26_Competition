#!/usr/bin/env bash
# One-shot fix: create .venv and install compatible packages.
# Run inside the GPU pod from repo root:
#   bash dsmlp/bootstrap_venv.sh
#
# Use this if install_env.sh is outdated or pip polluted ~/.local with numpy 2.x.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> GPU"
nvidia-smi | head -20

echo "==> Remove stale venv if broken"
rm -rf .venv

echo "==> Create venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> pip + numpy 1.x FIRST"
python -m pip install -U pip wheel
python -m pip install "numpy>=1.26.4,<2"
python -m pip install "scipy>=1.11,<1.15" "scikit-learn>=1.3,<1.6"

echo "==> transformers 4.x (NOT 5.x) + inference stack"
python -m pip install \
  "transformers>=4.51.0,<5.0" \
  "accelerate>=1.0.0,<2.0" \
  "bitsandbytes>=0.43.0,<0.50" \
  tqdm pandas pyyaml sympy "antlr4-python3-runtime==4.11"

echo "==> vllm (may upgrade numpy; we re-pin after)"
python -m pip install "vllm==0.8.5.post1"
python -m pip install "numpy>=1.26.4,<2"

echo "==> Verify imports (must use THIS python)"
python - <<'PY'
import numpy as np
import scipy
import sklearn
import transformers
import torch
print("python       =", __import__("sys").executable)
print("numpy        =", np.__version__)
print("transformers =", transformers.__version__)
print("cuda         =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       =", torch.cuda.get_device_name(0))
PY

echo ""
echo "SUCCESS. Always run:"
echo "  source $(pwd)/.venv/bin/activate"
echo "  python scripts/smoke_test_vllm.py --n 3"
