"""Final submission hyperparameters for Gradescope reproducibility.

All defaults here match the settings used for our Kaggle submission
(unified accuracy ~0.636). Verification re-runs should call
``run_inference()`` with no overrides, or only override checkpoint paths
when resuming an interrupted job.
"""

from __future__ import annotations

from typing import Any

from .vllm_runtime import MODEL_ID

# Hardware used for the reported submission runs (DSMLP).
GPU_TYPE = "NVIDIA A30 (24 GB VRAM, Ampere)"
APPROX_INFERENCE_WALL_TIME = "3–4 hours for 943 private questions (batch_size=64, includes retry/repair passes)"

# We did not fine-tune; verification loads the designated base model.
HF_MODEL_ID = MODEL_ID
HF_CACHE_DIR = "~/.cache/huggingface/hub"

FINAL_INFERENCE_DEFAULTS: dict[str, Any] = {
    "model_id": HF_MODEL_ID,
    "backend": "vllm",
    "test_path": "data/private.jsonl",
    "out_path": "submission.csv",
    "responses_jsonl": "results/submission/responses.jsonl",
    "batch_size": 64,
    "n_mcq": 5,
    "n_free": 3,
    "max_retries": 1,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 20,
    "seed": 0,
    "gpu_memory_utilization": 0.75,
    "max_model_len": 16384,
    "max_num_seqs": 96,
    "enforce_eager": True,
    "primary_prompt": "strict",
    "retry_prompt": "commit_now",
    "expected_rows": 943,
    "skip_inference": False,
    "resume_require_sane_boxed": False,
}


def merge_inference_kwargs(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return final defaults with optional overrides (None values ignored)."""
    out = dict(FINAL_INFERENCE_DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                out[key] = value
    return out
