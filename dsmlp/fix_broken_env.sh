#!/usr/bin/env bash
# Quick repair if you already ran the old install_env.sh and hit numpy/scipy errors.
# Run inside the GPU container from the repo root:
#   bash dsmlp/fix_broken_env.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No .venv found; run bash dsmlp/install_env.sh instead."
  exit 1
fi

python -m pip install "numpy>=1.26.4,<2" --force-reinstall
python -m pip install "scipy>=1.11,<1.15" "scikit-learn>=1.3,<1.6" --force-reinstall
python -m pip install "transformers>=4.51.0,<5.0" --force-reinstall

python -c "import numpy, scipy, sklearn, transformers; print('OK', numpy.__version__, transformers.__version__)"
