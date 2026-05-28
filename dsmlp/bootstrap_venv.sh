#!/usr/bin/env bash
# One-shot fix: create .venv and install compatible packages.
# Run inside the GPU pod from repo root:
#   bash dsmlp/bootstrap_venv.sh
#
# Uses PIP_NO_CACHE_DIR and a single vllm-first requirements install to avoid
# downloading torch twice (accelerate alone pulls ~3GB of CUDA wheels).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> GPU"
nvidia-smi | head -20

echo "==> Free pip cache (DSMLP home quota is tight)"
python3 -m pip cache purge 2>/dev/null || true
rm -rf .venv

echo "==> Create venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

export PIP_NO_CACHE_DIR=1

echo "==> numpy 1.x + scipy/sklearn (for transformers import chain)"
python -m pip install -U pip wheel
python -m pip install "numpy>=1.26.4,<2"
python -m pip install "scipy>=1.11,<1.15" "scikit-learn>=1.3,<1.6"

echo "==> vllm stack in ONE install (torch 2.6 only — do not pip install accelerate first)"
REQ="requirements-dsmlp.txt"
if [[ ! -f "$REQ" ]]; then
  echo "ERROR: $REQ missing. Run: git pull" >&2
  exit 1
fi
python -m pip install -r "$REQ"
python -m pip install "numpy>=1.26.4,<2"

echo "==> Verify imports (must use THIS python)"
python - <<'PY'
import sys
import numpy as np
import scipy
import sklearn
import transformers
import torch
print("python       =", sys.executable)
print("numpy        =", np.__version__)
print("transformers =", transformers.__version__)
print("torch        =", torch.__version__)
assert np.__version__.startswith("1."), np.__version__
assert transformers.__version__.startswith("4."), transformers.__version__
print("cuda         =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       =", torch.cuda.get_device_name(0))
PY

echo ""
echo "SUCCESS. Always run:"
echo "  source $(pwd)/.venv/bin/activate"
echo "  python scripts/smoke_test_vllm.py --n 3"
