#!/usr/bin/env bash
# Reclaim space on DSMLP before reinstalling .venv (home quota is often ~10–20 GB).
set -euo pipefail

echo "==> Disk usage (home)"
du -sh ~ 2>/dev/null || true
du -sh ~/.cache/pip ~/.local 2>/dev/null || true

echo "==> Purge pip download cache"
python3 -m pip cache purge 2>/dev/null || pip cache purge 2>/dev/null || true

echo "==> Remove broken project venv"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rm -rf .venv

echo "==> Optional: trim Hugging Face cache (re-downloads models on next run)"
# Uncomment if still over quota:
# rm -rf ~/.cache/huggingface/hub/*

echo "Done. Re-run: bash dsmlp/bootstrap_venv.sh"
