#!/usr/bin/env python3
"""Single entry point for building the private-set submission.

Calling ``run_inference()`` loads the configured model, runs batched
private-set inference with resumable checkpoints, applies the existing
self-consistency/retry/post-processing pipeline, and writes the final
``submission.csv``.
"""

from __future__ import annotations

from scripts.build_submission import main, run_inference

__all__ = ["run_inference"]


if __name__ == "__main__":
    raise SystemExit(main())
