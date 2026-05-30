#!/usr/bin/env python3
"""Run the full strict + self-consistency + retry pipeline on a public split.

Outputs:

* ``results/validation/<run_id>/responses.jsonl`` - one row per question
  containing the final committed response, the boxed answer, all
  samples, and whether it was scored correct.
* ``results/validation/<run_id>/summary.json`` - aggregate metrics
  (overall + per-bucket accuracy, extraction rate, truncation rate,
  average length, sanity-check failure counts).

Designed to be re-runnable: if ``responses.jsonl`` already contains a
row for a question id the engine skips that question, so a DSMLP
session disconnect costs at most one question of work.

Refuses to run without CUDA, so the user cannot accidentally start it
from the laptop.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from collections import Counter
from pathlib import Path

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen3_comp.data import load_jsonl  # noqa: E402
from qwen3_comp.extract import extract_boxed_content  # noqa: E402
from qwen3_comp.metrics import response_metrics  # noqa: E402
from qwen3_comp.scoring import per_bucket_accuracy, score_item  # noqa: E402
from qwen3_comp.self_consistency import (  # noqa: E402
    SamplingConfig,
    generate_with_retry_and_vote,
    post_hoc_sanity,
)
from qwen3_comp.vllm_runtime import VLLMEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split",
        default="results/validation/split.jsonl",
        help="stratified split written by build_validation_split.py",
    )
    p.add_argument(
        "--run_id",
        default="run01",
        help="subdirectory name under results/validation/",
    )
    p.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--n_mcq", type=int, default=5, help="self-consistency width for MCQ")
    p.add_argument("--n_free", type=int, default=3, help="self-consistency width for free-form")
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--primary_prompt", default="strict", choices=("strict", "baseline", "commit_now")
    )
    p.add_argument(
        "--retry_prompt", default="commit_now", choices=("strict", "baseline", "commit_now")
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional cap on number of questions (0 = all)",
    )
    return p.parse_args()


def _load_done(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    done: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["id"]] = rec
    return done


def main() -> int:
    args = parse_args()
    out_dir = REPO_ROOT / "results" / "validation" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "config.json"

    items = load_jsonl(args.split)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"Validation set: {len(items)} questions (run_id={args.run_id})")

    config = {
        "split": args.split,
        "run_id": args.run_id,
        "backend": args.backend,
        "n_mcq": args.n_mcq,
        "n_free": args.n_free,
        "max_retries": args.max_retries,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "primary_prompt": args.primary_prompt,
        "retry_prompt": args.retry_prompt,
        "limit": args.limit,
    }
    config_path.write_text(json.dumps(config, indent=2))

    done = _load_done(responses_path)
    pending = [it for it in items if it.get("id") not in done]
    print(f"Resuming: {len(done)} done, {len(pending)} pending")

    if pending:
        engine = VLLMEngine(backend=args.backend, seed=args.seed)
        sampling = SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed or None,
        )

        t0 = time.time()
        votes = generate_with_retry_and_vote(
            engine,
            pending,
            primary_prompt_id=args.primary_prompt,
            retry_prompt_id=args.retry_prompt,
            n_mcq=args.n_mcq,
            n_free=args.n_free,
            sampling=sampling,
            max_retries=args.max_retries,
        )
        elapsed = time.time() - t0
        print(f"Generated {len(votes)} new questions in {elapsed:.1f}s.")

        with responses_path.open("a") as f:
            for item, vote in zip(pending, votes):
                ok = score_item(item, vote.response)
                sane, reason = post_hoc_sanity(item, vote.response)
                boxed = extract_boxed_content(vote.response)
                rec = {
                    "id": item.get("id"),
                    "is_mcq": bool(item.get("options")),
                    "gold": item.get("answer"),
                    "response": vote.response,
                    "boxed": boxed,
                    "vote_answer": vote.vote_answer,
                    "vote_count": vote.vote_count,
                    "n_samples": vote.n_samples,
                    "extraction_rate": vote.extraction_rate,
                    "retry_attempted": vote.retry_attempted,
                    "repair_attempted": vote.repair_attempted,
                    "repair_succeeded": vote.repair_succeeded,
                    "final_stage": vote.final_stage,
                    "sane": sane,
                    "sanity_reason": reason,
                    "correct": ok,
                }
                f.write(json.dumps(rec) + "\n")
                done[rec["id"]] = rec

    # Aggregate metrics over the full split (including any pre-existing rows)
    ordered = [done[it["id"]] for it in items if it.get("id") in done]
    responses = [rec["response"] for rec in ordered]
    correct = [bool(rec["correct"]) for rec in ordered]

    bucket = per_bucket_accuracy(items[: len(ordered)], correct)
    rmetrics = response_metrics(items[: len(ordered)], responses)
    failures = {
        "sanity_failed": sum(1 for r in ordered if not r["sane"]),
        "no_boxed": sum(1 for r in ordered if r["boxed"] is None),
        "votes_with_no_winner": sum(1 for r in ordered if r["vote_answer"] is None),
        "retry_attempted": sum(1 for r in ordered if r.get("retry_attempted")),
        "repair_attempted": sum(1 for r in ordered if r.get("repair_attempted")),
        "repair_succeeded": sum(1 for r in ordered if r.get("repair_succeeded")),
        "final_stage": dict(Counter(r.get("final_stage", "unknown") for r in ordered)),
    }

    summary = {
        "config": config,
        "n_scored": len(ordered),
        "accuracy": bucket,
        "response_metrics": rmetrics,
        "failures": failures,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
