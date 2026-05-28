"""Local scoring that mirrors the official judger pipeline.

For MCQ we use strict letter extraction (no last-capital fallback) so
local accuracy stays a faithful proxy of the Kaggle evaluator. For
free-form we delegate to ``judger.Judger.auto_judge``.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from judger import Judger  # noqa: E402

from .extract import extract_mcq_letter
from .prompts import expected_num_answers


@lru_cache(maxsize=1)
def get_judger() -> Judger:
    return Judger(strict_extract=False)


def score_mcq(response: str, gold_letter: str) -> bool:
    pred = extract_mcq_letter(response)
    if pred is None:
        return False
    return pred == str(gold_letter).strip().upper()


def score_freeform(response: str, gold: list[str] | str) -> bool:
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return get_judger().auto_judge(
            pred=response,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        return False


def score_item(item: dict, response: str) -> bool:
    if item.get("options"):
        return score_mcq(response, str(item["answer"]))
    return score_freeform(response, item["answer"])


def score_batch(items: list[dict], responses: list[str]) -> list[bool]:
    if len(items) != len(responses):
        raise ValueError(
            f"items/responses length mismatch: {len(items)} vs {len(responses)}"
        )
    return [score_item(it, r) for it, r in zip(items, responses)]


def per_bucket_accuracy(
    items: list[dict], correct: list[bool]
) -> dict[str, dict[str, float]]:
    """Return overall/MCQ/free-form single/free-form multi accuracy."""
    counts = {"all": 0, "mcq": 0, "free_single": 0, "free_multi": 0}
    hits = {"all": 0, "mcq": 0, "free_single": 0, "free_multi": 0}
    for it, ok in zip(items, correct):
        counts["all"] += 1
        if ok:
            hits["all"] += 1
        if it.get("options"):
            bucket = "mcq"
        elif expected_num_answers(it) <= 1:
            bucket = "free_single"
        else:
            bucket = "free_multi"
        counts[bucket] += 1
        if ok:
            hits[bucket] += 1
    return {
        b: {
            "correct": hits[b],
            "total": counts[b],
            "accuracy": (hits[b] / counts[b]) if counts[b] else 0.0,
        }
        for b in counts
    }
