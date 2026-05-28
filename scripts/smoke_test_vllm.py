#!/usr/bin/env python3
"""DSMLP-only smoke test for the vLLM Qwen3-4B-Thinking pipeline.

Loads Qwen via vLLM, runs 1-3 public examples through the strict prompt,
prints whether each response contains an extractable ``\\boxed{...}``,
and scores them with the local judger as a sanity check.

Usage (inside a DSMLP GPU container):

    python scripts/smoke_test_vllm.py --n 3

Refuses to run on a machine without CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen3_comp.data import load_jsonl, stratify_split  # noqa: E402
from qwen3_comp.extract import extract_boxed_content  # noqa: E402
from qwen3_comp.scoring import score_item  # noqa: E402
from qwen3_comp.self_consistency import (  # noqa: E402
    generate_with_retry_and_vote,
)
from qwen3_comp.vllm_runtime import VLLMEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/public.jsonl")
    p.add_argument("--n", type=int, default=3, help="total smoke-test questions")
    p.add_argument(
        "--backend",
        choices=("vllm", "transformers"),
        default="vllm",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    items = load_jsonl(args.data)
    n_each = max(1, args.n // 3)
    subset = stratify_split(
        items,
        n_mcq=n_each,
        n_free_single=n_each,
        n_free_multi=max(0, args.n - 2 * n_each),
        seed=args.seed,
    )
    print(f"Smoke test on {len(subset)} questions ({args.backend} backend).")

    engine = VLLMEngine(backend=args.backend)
    t0 = time.time()
    results = generate_with_retry_and_vote(
        engine,
        subset,
        primary_prompt_id="strict",
        retry_prompt_id="commit_now",
        n_mcq=2,
        n_free=2,
        max_retries=1,
    )
    elapsed = time.time() - t0
    print(f"\nGenerated {len(results)} questions in {elapsed:.1f}s.\n")

    n_boxed = 0
    n_correct = 0
    for item, res in zip(subset, results):
        boxed = extract_boxed_content(res.response)
        ok = score_item(item, res.response)
        if boxed is not None:
            n_boxed += 1
        if ok:
            n_correct += 1
        print(
            f"id={item.get('id')} "
            f"type={'MCQ' if item.get('options') else 'free'} "
            f"vote={res.vote_answer!r} ({res.vote_count}/{res.n_samples}) "
            f"boxed={boxed!r} "
            f"gold={item.get('answer')!r} "
            f"correct={ok}"
        )

    print(
        f"\nSummary: boxed {n_boxed}/{len(results)}, correct "
        f"{n_correct}/{len(results)}."
    )
    if n_boxed == 0:
        print(
            "WARNING: 0 boxed answers. Check prompts/budgets before running "
            "the full validation."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
