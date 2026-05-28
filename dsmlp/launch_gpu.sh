#!/usr/bin/env bash
# Convenience wrapper for launching a DSMLP GPU container suitable for
# Qwen3-4B-Thinking inference.
#
# Usage (from your laptop, after `ssh dsmlp-login.ucsd.edu`):
#
#     bash dsmlp/launch_gpu.sh                # 1 GPU, 8 CPU, 32 GB, foreground
#     bash dsmlp/launch_gpu.sh -- nvidia-smi  # one-shot command
#     bash dsmlp/launch_gpu.sh -b             # 6h background pod
#     bash dsmlp/launch_gpu.sh -B -- bash -lc 'python scripts/smoke_test_vllm.py'
#
# DO NOT run training/inference directly on dsmlp-login. This script
# starts a GPU pod with the right resources; everything else runs
# inside that pod.

set -euo pipefail

LAUNCHER="${LAUNCHER:-launch-scipy-ml.sh}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM_GB="${MEM_GB:-32}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-43200}"  # 12 hours

if ! command -v "$LAUNCHER" >/dev/null 2>&1; then
  echo "ERROR: $LAUNCHER not found in PATH." >&2
  echo "  Are you logged into dsmlp-login.ucsd.edu?" >&2
  exit 1
fi

export K8S_TIMEOUT_SECONDS="$TIMEOUT_SECONDS"

exec "$LAUNCHER" -g "$GPUS" -c "$CPUS" -m "$MEM_GB" "$@"
