"""Boxed-answer extraction and format validation.

This is a thin layer on top of the official ``utils.last_boxed_only_string``
extractor so our retry policy and metrics agree with the evaluator.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import last_boxed_only_string  # noqa: E402


_MCQ_LETTER_RE = re.compile(r"^[A-J]$")


def extract_boxed_content(response: str) -> Optional[str]:
    """Return the inner content of the last ``\\boxed{...}`` in *response*.

    Returns ``None`` if the response has no complete boxed group. Matches
    the behaviour the judger relies on, including balanced-brace
    handling for nested expressions.
    """
    if not response:
        return None
    boxed_substr = last_boxed_only_string(response)
    if boxed_substr is None:
        return None
    if boxed_substr.startswith("\\boxed{") and boxed_substr.endswith("}"):
        return boxed_substr[len("\\boxed{") : -1].strip()
    if boxed_substr.startswith("\\fbox{") and boxed_substr.endswith("}"):
        return boxed_substr[len("\\fbox{") : -1].strip()
    return None


def split_box_subanswers(inner: str) -> list[str]:
    """Split the inside of a ``\\boxed{...}`` into ordered sub-answers.

    The splitter is bracket-aware so that LaTeX expressions like
    ``\\frac{a, b}{c}`` are not torn apart on the inner comma.
    """
    if inner is None:
        return []
    out: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            out.append(inner[start:i].strip())
            start = i + 1
    if start < len(inner):
        out.append(inner[start:].strip())
    return [p for p in out if p]


def extract_mcq_letter(response: str) -> Optional[str]:
    """Return the boxed MCQ letter (A-J) if present, else ``None``.

    Unlike the starter notebook helper we never fall back to "last
    standalone capital letter", because that heuristic credits or
    penalises responses based on incidental capitals in the reasoning.
    """
    inner = extract_boxed_content(response)
    if not inner:
        return None
    letter = inner.strip().upper()
    if len(letter) == 1 and _MCQ_LETTER_RE.match(letter):
        return letter
    return None


def is_valid_response(
    response: str,
    *,
    is_mcq: bool,
    expected_num: int = 1,
) -> tuple[bool, str]:
    """Quick local sanity check used by the retry policy.

    Returns ``(ok, reason)``. ``ok`` is True only if the response
    contains exactly one extractable final ``\\boxed{...}`` whose
    content matches the expected shape:

    * MCQ: a single letter A-J.
    * Free-form: ``expected_num`` comma-separated sub-answers.
    """
    if not response or not response.strip():
        return False, "empty_response"
    inner = extract_boxed_content(response)
    if inner is None:
        return False, "no_boxed_answer"
    if is_mcq:
        letter = extract_mcq_letter(response)
        if not letter:
            return False, "boxed_not_letter"
        return True, "ok"
    parts = split_box_subanswers(inner)
    if not parts:
        return False, "empty_boxed_answer"
    if expected_num > 1 and len(parts) != expected_num:
        return False, f"wrong_subanswer_count:{len(parts)}!={expected_num}"
    return True, "ok"
