#!/usr/bin/env python3
"""Gradescope single entry point for the CSE 151B competition submission.

Calling ``run_inference()`` (or ``python run_inference.py``) runs the
complete pipeline end-to-end on the private set and writes ``submission.csv``.

Nothing else is required: model loading, generation, self-consistency,
sanity-triggered retry/repair, and CSV export are all handled inside
``run_inference()``.

Example (DSMLP GPU pod)::

    source .venv/bin/activate
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    python run_inference.py

Example (Python API)::

    from run_inference import run_inference
    report = run_inference()
    assert report["ok"]

See README.md for environment setup, model weights, and hyperparameters.
"""

from __future__ import annotations

from scripts.build_submission import main, run_inference

__all__ = ["run_inference"]


if __name__ == "__main__":
    raise SystemExit(main())
