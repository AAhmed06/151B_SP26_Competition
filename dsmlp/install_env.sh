#!/usr/bin/env bash
# Install into project .venv (never ~/.local). Run inside GPU pod only.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Launch with: launch-scipy-ml.sh -g 1 -c 8 -m 32 -v a30" >&2
  exit 1
fi

echo "==> nvidia-smi"
nvidia-smi | head -25

if nvidia-smi -L 2>/dev/null | grep -qiE 'MIG|Blackwell'; then
  echo "WARNING: MIG/Blackwell detected. Prefer: launch-scipy-ml.sh -g 1 -c 8 -m 32 -v a30"
fi

echo "==> Creating project venv"
rm -rf .venv
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python3 -m pip cache purge 2>/dev/null || true
export PIP_NO_CACHE_DIR=1

python -m pip install -U pip wheel
python -m pip install "numpy>=1.26.4,<2"
python -m pip install "scipy>=1.11,<1.15" "scikit-learn>=1.3,<1.6"

if [[ -f requirements-dsmlp.txt ]]; then
  python -m pip install -r requirements-dsmlp.txt
else
  echo "ERROR: requirements-dsmlp.txt missing; git pull or use bootstrap_venv.sh" >&2
  exit 1
fi

python -m pip install "numpy>=1.26.4,<2"

python - <<'PY'
import sys
import numpy as np
import scipy, sklearn, transformers, torch
print("venv python  =", sys.executable)
print("numpy       =", np.__version__)
print("transformers=", transformers.__version__)
assert np.__version__.startswith("1."), f"need numpy 1.x, got {np.__version__}"
assert transformers.__version__.startswith("4."), f"need transformers 4.x, got {transformers.__version__}"
print("cuda        =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      =", torch.cuda.get_device_name(0))
PY

echo ""
echo "Done. Run:"
echo "  source $REPO_ROOT/.venv/bin/activate"
echo "  python scripts/smoke_test_vllm.py --n 3"
