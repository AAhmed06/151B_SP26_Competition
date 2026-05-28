#!/usr/bin/env bash
# Install the Python dependencies needed for vLLM Qwen3-4B inference
# *inside* a DSMLP GPU container. Run after launching the container,
# never on dsmlp-login.

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Are you inside a -g 1 container?" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"

echo "==> nvidia-smi"
nvidia-smi || true

echo "==> Installing pinned wheels"
"$PYTHON" -m pip install --upgrade pip wheel

# vLLM 0.8.5.post1 matches the starter notebook's Triton/CUDA expectations.
# bitsandbytes provides on-the-fly 4-bit quantisation for the fallback path.
"$PYTHON" -m pip install \
  "vllm==0.8.5.post1" \
  "transformers>=4.51" \
  "accelerate>=1.0" \
  "bitsandbytes>=0.43" \
  "tqdm" \
  "pandas" \
  "pyyaml" \
  "sympy" \
  "antlr4-python3-runtime==4.11" \
  "datasets>=2.20" \
  "peft>=0.11"

echo "==> Verifying CUDA visibility"
"$PYTHON" - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count   =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name    =", torch.cuda.get_device_name(0))
PY
