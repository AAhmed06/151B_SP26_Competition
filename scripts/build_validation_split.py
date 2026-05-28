#!/usr/bin/env python3
"""Build a deterministic stratified validation split from data/public.jsonl.

The split is stored as a JSONL of full items so downstream scripts can
pick it up without re-stratifying. Saves both the split items and a
summary of bucket counts.

This script is CPU-only (no GPU required), so it can be run from your
laptop or from DSMLP. The output is committed via the run artifact, not
the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen3_comp.data import (  # noqa: E402
    load_jsonl,
    stratify_split,
    summarise_split,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/public.jsonl")
    p.add_argument(
        "--out",
        default="results/validation/split.jsonl",
        help="output JSONL containing the stratified items",
    )
    p.add_argument("--n_mcq", type=int, default=50)
    p.add_argument("--n_free_single", type=int, default=80)
    p.add_argument("--n_free_multi", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    items = load_jsonl(args.data)
    print(f"Loaded {len(items)} items from {args.data}")

    full_summary = summarise_split(items)
    print("Full distribution:", json.dumps(full_summary))

    subset = stratify_split(
        items,
        n_mcq=args.n_mcq,
        n_free_single=args.n_free_single,
        n_free_multi=args.n_free_multi,
        seed=args.seed,
    )
    summary = summarise_split(subset)
    print("Validation subset:", json.dumps(summary))

    write_jsonl(args.out, subset)
    print(f"Wrote {len(subset)} items to {args.out}")

    meta_path = Path(args.out).with_suffix(".meta.json")
    meta = {
        "source": str(args.data),
        "seed": args.seed,
        "n_requested": {
            "mcq": args.n_mcq,
            "free_single": args.n_free_single,
            "free_multi": args.n_free_multi,
        },
        "actual": summary,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
