"""Token budget routing.

Section "Phase 4" of the plan: do not use one fixed ``max_new_tokens`` for
every item. Route the budget by question type, question length, and retry
status so that simple questions do not waste KV-cache memory and hard
questions do not get truncated.
"""

from __future__ import annotations

from .prompts import expected_num_answers


def estimate_question_difficulty(item: dict) -> str:
    """Return ``"short"``, ``"medium"``, or ``"long"`` for routing."""
    question = item.get("question", "")
    q_len = len(question)
    options = item.get("options") or []
    options_text_len = sum(len(str(o)) for o in options)
    total_chars = q_len + options_text_len

    if total_chars > 1200:
        return "long"
    if total_chars > 400:
        return "medium"
    return "short"


def route_max_new_tokens(item: dict, *, retry: int = 0) -> int:
    """Choose ``max_new_tokens`` for *item* at retry attempt ``retry``.

    The retry==0 budget is sized so the model can comfortably produce
    its chain-of-thought and a final boxed answer. retry>=1 bumps the
    budget upwards because the only reason we retry is "no boxed
    answer", which the milestone report attributes to truncation.
    """
    diff = estimate_question_difficulty(item)
    is_mcq = bool(item.get("options"))
    n_ans = expected_num_answers(item)

    if retry >= 1:
        return 8192 if diff == "long" else 6144

    if is_mcq:
        if diff == "long":
            return 4096
        if diff == "medium":
            return 3072
        return 2048

    if n_ans >= 3 or diff == "long":
        return 4096
    if diff == "medium":
        return 3072
    return 2048
