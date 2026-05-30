"""Per-run diagnostics used alongside accuracy.

The milestone report showed that score regressions are usually caused
by truncation (no final boxed answer) rather than by wrong reasoning.
Tracking these metrics for every validation run makes that failure
mode visible at a glance.
"""

from __future__ import annotations

from .extract import extract_boxed_content, split_box_subanswers
from .prompts import expected_num_answers


def looks_truncated(response: str) -> bool:
    """Heuristic: response appears to have been cut off mid-thought.

    Triggers when the response ends inside a Qwen ``<think>`` block or
    finishes without a closing ``\\boxed{...}``. False positives are
    acceptable; we only use this for diagnostics, never for scoring.
    """
    if not response:
        return True
    tail = response.rstrip()[-200:]
    if "<think>" in tail and "</think>" not in tail:
        return True
    if extract_boxed_content(response) is None:
        return True
    return False


def response_metrics(items: list[dict], responses: list[str]) -> dict:
    """Aggregate per-response diagnostics for a (items, responses) run."""
    total = len(items)
    if total == 0:
        return {
            "total": 0,
            "boxed_rate": 0.0,
            "trunc_rate": 0.0,
            "avg_chars": 0.0,
            "avg_tail_chars": 0.0,
            "mismatched_subanswer_count": 0,
        }

    boxed = 0
    trunc = 0
    mismatched = 0
    total_chars = 0
    for it, resp in zip(items, responses):
        resp = resp or ""
        total_chars += len(resp)
        inner = extract_boxed_content(resp)
        if inner is not None:
            boxed += 1
        if looks_truncated(resp):
            trunc += 1
        if not it.get("options") and inner is not None:
            expected = expected_num_answers(it)
            if expected > 1:
                parts = split_box_subanswers(inner)
                if len(parts) != expected:
                    mismatched += 1

    return {
        "total": total,
        "boxed_rate": boxed / total,
        "trunc_rate": trunc / total,
        "avg_chars": total_chars / total,
        "mismatched_subanswer_count": mismatched,
    }
