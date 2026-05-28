"""Dataset loading and stratified split construction."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from .prompts import expected_num_answers


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    items: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, items: Iterable[dict]) -> None:
    """Write an iterable of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def is_mcq(item: dict) -> bool:
    return bool(item.get("options"))


def stratify_split(
    items: list[dict],
    *,
    n_mcq: int,
    n_free_single: int,
    n_free_multi: int,
    seed: int = 42,
) -> list[dict]:
    """Return a stratified subset with the requested per-bucket counts.

    Buckets: MCQ, free-form single-[ANS], free-form multi-[ANS]. Items
    are sampled without replacement from each bucket; if a bucket has
    fewer items than requested the whole bucket is used.
    """
    rng = random.Random(seed)

    mcq_pool: list[dict] = []
    free_single: list[dict] = []
    free_multi: list[dict] = []

    for it in items:
        if is_mcq(it):
            mcq_pool.append(it)
            continue
        n = expected_num_answers(it)
        if n <= 1:
            free_single.append(it)
        else:
            free_multi.append(it)

    def pick(pool: list[dict], k: int) -> list[dict]:
        if k <= 0:
            return []
        if k >= len(pool):
            return list(pool)
        return rng.sample(pool, k)

    subset: list[dict] = []
    subset.extend(pick(mcq_pool, n_mcq))
    subset.extend(pick(free_single, n_free_single))
    subset.extend(pick(free_multi, n_free_multi))
    rng.shuffle(subset)
    return subset


def summarise_split(items: list[dict]) -> dict:
    """Return per-bucket counts for logging/reporting."""
    n_mcq = sum(1 for it in items if is_mcq(it))
    n_free_single = 0
    n_free_multi = 0
    for it in items:
        if is_mcq(it):
            continue
        if expected_num_answers(it) <= 1:
            n_free_single += 1
        else:
            n_free_multi += 1
    return {
        "total": len(items),
        "mcq": n_mcq,
        "free_single": n_free_single,
        "free_multi": n_free_multi,
    }
