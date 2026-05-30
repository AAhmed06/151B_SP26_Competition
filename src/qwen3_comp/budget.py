"""Token budget routing.

Section "Phase 4" of the plan: do not use one fixed ``max_new_tokens`` for
every item. Route the budget by question type, question length, and retry
status so that simple questions do not waste KV-cache memory and hard
questions do not get truncated.
"""

from __future__ import annotations

import re

from .prompts import expected_num_answers

# Patterns that indicate hard multi-step reasoning (penalised with larger budget)
_HARD_MATH_RE = re.compile(
    r"prove|probability|combinat|integral|deriv|matrix|determinant"
    r"|eigenvalue|sequence|series|recur|modulo|congruent|polynomial"
    r"|maximize|minimize|optimization|triangle|circle|polygon",
    re.IGNORECASE,
)


def estimate_question_difficulty(item: dict) -> str:
    """Return ``"short"``, ``"medium"``, or ``"long"`` for routing.

    Combines text length with keyword heuristics: questions containing
    multi-step reasoning keywords are bumped at least to "medium" so
    the thinking model has enough budget to finish its chain-of-thought.
    """
    question = item.get("question", "")
    q_len = len(question)
    options = item.get("options") or []
    options_text_len = sum(len(str(o)) for o in options)
    total_chars = q_len + options_text_len

    if total_chars > 1000:
        return "long"
    if total_chars > 350:
        tier = "medium"
    else:
        tier = "short"

    # Bump short → medium when the question text contains hard-math keywords
    if tier == "short" and _HARD_MATH_RE.search(question):
        tier = "medium"

    return tier


def route_max_new_tokens(item: dict, *, retry: int = 0) -> int:
    """Choose ``max_new_tokens`` for *item* at retry attempt ``retry``.

    Budgets are deliberately generous because Qwen3-4B-Thinking emits
    long ``<think>`` traces before the final ``\\boxed{}``.  The #1
    failure mode in our runs is truncation before the boxed answer, so
    we trade some throughput for a higher boxed-answer rate.

    Primary budgets (retry=0):
      MCQ  short/medium/long : 4096 / 6144 / 8192
      Free short/medium/long : 4096 / 6144 / 8192 (capped by n_ans)

    Retry budgets (retry>=1):
      long : 12288  (max safe given max_model_len=16384 and ~500-tok prompt)
      else : 8192
    """
    diff = estimate_question_difficulty(item)
    is_mcq = bool(item.get("options"))
    n_ans = expected_num_answers(item)

    if retry >= 1:
        return 12288 if diff == "long" else 8192

    if is_mcq:
        if diff == "long":
            return 8192
        if diff == "medium":
            return 6144
        return 4096

    # Free-form: multi-answer questions always get the long budget
    if n_ans >= 3 or diff == "long":
        return 8192
    if diff == "medium":
        return 6144
    return 4096
