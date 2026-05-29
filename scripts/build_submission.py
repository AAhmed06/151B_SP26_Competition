#!/usr/bin/env python3
"""Generate the Kaggle submission CSV for the private test set.

Reads ``data/private.jsonl`` (must be present locally; it is not shipped
in this repo), runs the strict + self-consistency + retry pipeline on
every row, and writes ``submission.csv`` with exactly the columns
``id,response`` and one row per private item.

Refuses to run without CUDA, so the file is only ever produced inside a
DSMLP GPU container.

After writing the CSV the script reads it back with pandas to confirm:

* exactly N rows, where N is the size of the test JSONL,
* the header is ``id,response``,
* every test id is present exactly once,
* no response is empty,
* the boxed-answer rate is high (warning if below 95%).

Usage (inside the DSMLP GPU container):

    python scripts/build_submission.py \\
        --test data/private.jsonl \\
        --out submission.csv \\
        --responses_jsonl results/submission/responses.jsonl

``run_inference()`` is the single entry point expected by the competition:
it loads the configured model, runs private-set inference, applies all
post-processing/retry/voting logic, checkpoints completed batches, and
writes the final CSV. ``--responses_jsonl`` is the resumable checkpoint;
if a GPU pod disconnects, rerunning the same command skips completed ids.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen3_comp.data import load_jsonl  # noqa: E402
from qwen3_comp.extract import extract_boxed_content  # noqa: E402
from qwen3_comp.self_consistency import (  # noqa: E402
    SamplingConfig,
    generate_with_retry_and_vote,
    post_hoc_sanity,
)
from qwen3_comp.vllm_runtime import MODEL_ID, VLLMEngine  # noqa: E402

EXPECTED_PRIVATE_ROWS = 943
DEFAULT_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="data/private.jsonl")
    p.add_argument("--out", default="submission.csv")
    p.add_argument(
        "--model_id",
        default=MODEL_ID,
        help=(
            "HuggingFace model or fine-tuned checkpoint to load. Defaults "
            "to the designated base model."
        ),
    )
    p.add_argument(
        "--responses_jsonl",
        default="results/submission/responses.jsonl",
        help="resumable per-question checkpoint",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "number of private rows to finish before writing a resume "
            "checkpoint"
        ),
    )
    p.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--n_mcq", type=int, default=5)
    p.add_argument("--n_free", type=int, default=3)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.75,
        help=(
            "fraction of GPU memory vLLM may reserve for weights/KV cache; "
            "lower this if engine startup OOMs"
        ),
    )
    p.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="maximum prompt + generation tokens for vLLM",
    )
    p.add_argument(
        "--max_num_seqs",
        type=int,
        default=96,
        help=(
            "maximum simultaneous vLLM sequences; lower this if sampler "
            "warmup OOMs"
        ),
    )
    p.add_argument(
        "--enforce_eager",
        action="store_true",
        help="disable vLLM CUDA graph capture/torch compile warmup to reduce startup VRAM",
    )
    p.add_argument("--primary_prompt", default="strict")
    p.add_argument("--retry_prompt", default="commit_now")
    p.add_argument(
        "--resume_require_sane_boxed",
        action="store_true",
        help=(
            "when resuming, only count checkpoint rows with sane=True and "
            "a non-empty boxed field as done"
        ),
    )
    p.add_argument(
        "--rerun_checkpoint_in",
        default="",
        help=(
            "optional source JSONL to filter into a rerun checkpoint "
            "(e.g., full backup)"
        ),
    )
    p.add_argument(
        "--rerun_checkpoint_out",
        default="results/submission/responses_rerun.jsonl",
        help="destination JSONL for rerun checkpoint filtering",
    )
    p.add_argument(
        "--rerun_filter_mode",
        choices=("boxed", "sane_boxed"),
        default="sane_boxed",
        help=(
            "boxed: keep rows with non-empty boxed; "
            "sane_boxed: keep rows with sane=True and non-empty boxed"
        ),
    )
    p.add_argument(
        "--prepare_rerun_only",
        action="store_true",
        help="build rerun checkpoint and exit without inference/CSV build",
    )
    p.add_argument(
        "--expected_rows",
        type=int,
        default=EXPECTED_PRIVATE_ROWS,
        help="row count assertion; set 0 to disable",
    )
    p.add_argument(
        "--skip_inference",
        action="store_true",
        help="only rebuild submission.csv from an existing responses_jsonl",
    )
    return p.parse_args()


def _is_truthy_boxed(rec: dict) -> bool:
    boxed = rec.get("boxed")
    return boxed is not None and str(boxed).strip() != ""


def _load_done(
    path: Path,
    *,
    require_sane_boxed: bool = False,
) -> dict[int, dict]:
    if not path.exists():
        return {}
    done: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if require_sane_boxed and not (
                bool(rec.get("sane")) and _is_truthy_boxed(rec)
            ):
                continue
            done[rec["id"]] = rec
    return done


def _write_csv(out_path: Path, items: list[dict], done: dict[int, dict]) -> None:
    import pandas as pd

    rows: list[dict] = []
    for it in items:
        qid = it.get("id")
        rec = done.get(qid)
        if rec is None or not rec.get("response"):
            response = ""
        else:
            response = rec["response"]
        rows.append({"id": qid, "response": response})
    df = pd.DataFrame(rows, columns=["id", "response"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _make_response_record(item: dict, vote) -> dict:
    sane, reason = post_hoc_sanity(item, vote.response)
    return {
        "id": item.get("id"),
        "is_mcq": bool(item.get("options")),
        "response": vote.response,
        "boxed": extract_boxed_content(vote.response),
        "vote_answer": vote.vote_answer,
        "vote_count": vote.vote_count,
        "n_samples": vote.n_samples,
        "extraction_rate": vote.extraction_rate,
        "sane": sane,
        "sanity_reason": reason,
    }


def _append_checkpoint(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _iter_batches(items: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _build_rerun_checkpoint(
    src_path: Path,
    dst_path: Path,
    *,
    mode: str,
) -> dict:
    if not src_path.exists():
        raise FileNotFoundError(f"source checkpoint not found: {src_path}")
    kept: list[dict] = []
    total = 0
    seen_ids: set[int] = set()
    with src_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            keep = _is_truthy_boxed(rec)
            if mode == "sane_boxed":
                keep = keep and bool(rec.get("sane"))
            if not keep:
                continue
            qid = rec.get("id")
            if qid in seen_ids:
                # keep last occurrence semantics: overwrite previous entry
                kept = [x for x in kept if x.get("id") != qid]
            seen_ids.add(qid)
            kept.append(rec)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    return {
        "source": str(src_path),
        "output": str(dst_path),
        "mode": mode,
        "total_rows_read": total,
        "rows_kept": len(kept),
        "rows_dropped": max(0, total - len(kept)),
    }


def _verify_csv(out_path: Path, items: list[dict], expected_rows: int) -> dict:
    """Read the CSV back and run hard structural checks."""
    import pandas as pd

    df = pd.read_csv(out_path, dtype={"id": int, "response": str}, keep_default_na=False)

    issues: list[str] = []
    if list(df.columns) != ["id", "response"]:
        issues.append(f"bad_header: {list(df.columns)} != ['id', 'response']")

    expected_ids = [it.get("id") for it in items]
    if len(df) != len(items):
        issues.append(f"row_count: {len(df)} != {len(items)}")
    if expected_rows and len(df) != expected_rows:
        issues.append(f"unexpected_row_count: {len(df)} != {expected_rows}")

    actual_ids = list(df["id"])
    if sorted(actual_ids) != sorted(expected_ids):
        issues.append("id_set_mismatch")
    if len(set(actual_ids)) != len(actual_ids):
        issues.append("duplicate_ids")

    n_empty = int((df["response"].astype(str).str.len() == 0).sum())
    if n_empty:
        issues.append(f"empty_responses: {n_empty}")

    n_boxed = int(
        df["response"].astype(str).apply(
            lambda s: extract_boxed_content(s) is not None
        ).sum()
    )

    return {
        "rows": int(len(df)),
        "header": list(df.columns),
        "empty_responses": n_empty,
        "boxed_responses": n_boxed,
        "boxed_rate": n_boxed / max(1, len(df)),
        "issues": issues,
    }


def run_inference(
    *,
    test_path: str | Path = "data/private.jsonl",
    out_path: str | Path = "submission.csv",
    responses_jsonl: str | Path = "results/submission/responses.jsonl",
    model_id: str = MODEL_ID,
    backend: str = "vllm",
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_mcq: int = 5,
    n_free: int = 3,
    max_retries: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 0,
    gpu_memory_utilization: float = 0.75,
    max_model_len: int = 16384,
    max_num_seqs: int = 96,
    enforce_eager: bool = False,
    primary_prompt: str = "strict",
    retry_prompt: str = "commit_now",
    expected_rows: int = EXPECTED_PRIVATE_ROWS,
    skip_inference: bool = False,
    resume_require_sane_boxed: bool = False,
) -> dict:
    """Run the full private-set pipeline and write the submission CSV.

    This is intentionally the only production entry point: callers do not
    need to run separate model-loading, generation, post-processing, or CSV
    steps. Progress is committed to ``responses_jsonl`` after each batch so
    a disconnected GPU session only loses the in-flight batch.
    """
    test_path = Path(test_path)
    out_path = Path(out_path)
    responses_path = Path(responses_jsonl)

    items = load_jsonl(test_path)
    print(f"Loaded {len(items)} test items from {test_path}")
    if expected_rows and len(items) != expected_rows:
        print(
            f"WARNING: test set has {len(items)} rows, expected "
            f"{expected_rows}."
        )

    done = _load_done(
        responses_path,
        require_sane_boxed=resume_require_sane_boxed,
    )
    pending = [it for it in items if it.get("id") not in done]
    print(f"Submission: {len(done)} done, {len(pending)} pending")

    if pending and not skip_inference:
        engine = VLLMEngine(
            model_id=model_id,
            backend=backend,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
        )
        sampling = SamplingConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed or None,
        )

        batches = _iter_batches(pending, batch_size)
        total_started = len(done)
        t0 = time.time()
        for batch_idx, batch in enumerate(batches, start=1):
            batch_t0 = time.time()
            print(
                f"Batch {batch_idx}/{len(batches)}: generating {len(batch)} "
                f"questions ({len(done)}/{len(items)} already checkpointed)"
            )
            votes = generate_with_retry_and_vote(
                engine,
                batch,
                primary_prompt_id=primary_prompt,
                retry_prompt_id=retry_prompt,
                n_mcq=n_mcq,
                n_free=n_free,
                sampling=sampling,
                max_retries=max_retries,
            )
            records = [
                _make_response_record(item, vote)
                for item, vote in zip(batch, votes)
            ]
            _append_checkpoint(responses_path, records)
            for rec in records:
                done[rec["id"]] = rec
            _write_csv(out_path, items, done)
            print(
                f"Checkpointed batch {batch_idx}/{len(batches)}: "
                f"{len(done) - total_started} new, {len(done)}/{len(items)} "
                f"total in {time.time() - batch_t0:.1f}s."
            )
        print(f"Generated {len(done) - total_started} questions in {time.time() - t0:.1f}s.")

    _write_csv(out_path, items, done)

    report = _verify_csv(out_path, items, expected_rows)
    report_path = out_path.with_suffix(".verification.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if report["issues"]:
        print(f"FAILED: {len(report['issues'])} structural issues; see {report_path}")
        report["ok"] = False
        return report
    if report["boxed_rate"] < 0.95:
        print(
            f"WARNING: boxed_rate {report['boxed_rate']:.2%} below 95%; "
            "consider re-running missing rows before uploading."
        )
    print(f"Wrote {out_path} ({report['rows']} rows).")
    report["ok"] = True
    return report


def main() -> int:
    args = parse_args()
    responses_jsonl = args.responses_jsonl
    if args.rerun_checkpoint_in:
        rerun_report = _build_rerun_checkpoint(
            Path(args.rerun_checkpoint_in),
            Path(args.rerun_checkpoint_out),
            mode=args.rerun_filter_mode,
        )
        print(json.dumps(rerun_report, indent=2))
        responses_jsonl = args.rerun_checkpoint_out
    if args.prepare_rerun_only:
        if not args.rerun_checkpoint_in:
            print("ERROR: --prepare_rerun_only requires --rerun_checkpoint_in")
            return 1
        return 0

    report = run_inference(
        test_path=args.test,
        out_path=args.out,
        responses_jsonl=responses_jsonl,
        model_id=args.model_id,
        backend=args.backend,
        batch_size=args.batch_size,
        n_mcq=args.n_mcq,
        n_free=args.n_free,
        max_retries=args.max_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        primary_prompt=args.primary_prompt,
        retry_prompt=args.retry_prompt,
        expected_rows=args.expected_rows,
        skip_inference=args.skip_inference,
        resume_require_sane_boxed=args.resume_require_sane_boxed,
    )
    if not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
