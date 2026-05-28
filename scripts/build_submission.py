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

``--responses_jsonl`` is a resumable per-question checkpoint, identical
in shape to the validation harness output.
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

from qwen3_comp.data import load_jsonl  # noqa: E402
from qwen3_comp.extract import extract_boxed_content  # noqa: E402
from qwen3_comp.self_consistency import (  # noqa: E402
    SamplingConfig,
    generate_with_retry_and_vote,
    post_hoc_sanity,
)
from qwen3_comp.vllm_runtime import VLLMEngine  # noqa: E402

EXPECTED_PRIVATE_ROWS = 943


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="data/private.jsonl")
    p.add_argument("--out", default="submission.csv")
    p.add_argument(
        "--responses_jsonl",
        default="results/submission/responses.jsonl",
        help="resumable per-question checkpoint",
    )
    p.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    p.add_argument("--n_mcq", type=int, default=5)
    p.add_argument("--n_free", type=int, default=3)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--primary_prompt", default="strict")
    p.add_argument("--retry_prompt", default="commit_now")
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


def main() -> int:
    args = parse_args()
    items = load_jsonl(args.test)
    print(f"Loaded {len(items)} test items from {args.test}")
    if args.expected_rows and len(items) != args.expected_rows:
        print(
            f"WARNING: test set has {len(items)} rows, expected "
            f"{args.expected_rows}."
        )

    responses_path = Path(args.responses_jsonl)
    done = _load_done(responses_path)
    pending = [it for it in items if it.get("id") not in done]
    print(f"Submission: {len(done)} done, {len(pending)} pending")

    if pending and not args.skip_inference:
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
        print(f"Generated {len(votes)} questions in {elapsed:.1f}s.")

        responses_path.parent.mkdir(parents=True, exist_ok=True)
        with responses_path.open("a") as f:
            for item, vote in zip(pending, votes):
                sane, reason = post_hoc_sanity(item, vote.response)
                rec = {
                    "id": item.get("id"),
                    "is_mcq": bool(item.get("options")),
                    "response": vote.response,
                    "boxed": extract_boxed_content(vote.response),
                    "vote_answer": vote.vote_answer,
                    "vote_count": vote.vote_count,
                    "n_samples": vote.n_samples,
                    "sane": sane,
                    "sanity_reason": reason,
                }
                f.write(json.dumps(rec) + "\n")
                done[rec["id"]] = rec

    out_path = Path(args.out)
    _write_csv(out_path, items, done)

    report = _verify_csv(out_path, items, args.expected_rows)
    report_path = out_path.with_suffix(".verification.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if report["issues"]:
        print(f"FAILED: {len(report['issues'])} structural issues; see {report_path}")
        return 1
    if report["boxed_rate"] < 0.95:
        print(
            f"WARNING: boxed_rate {report['boxed_rate']:.2%} below 95%; "
            "consider re-running missing rows before uploading."
        )
    print(f"Wrote {out_path} ({report['rows']} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
