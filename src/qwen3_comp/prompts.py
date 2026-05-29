"""System and user prompt construction for the math competition.

Default behaviour follows the milestone report finding: strict-format prompts
maximise the chance that a final ``\\boxed{...}`` exists, which is the only
form that the official ``judger.py`` extracts reliably.
"""

from __future__ import annotations

from typing import Optional


SYSTEM_PROMPT_STRICT_MATH = (
    "You are an expert mathematician. Solve the problem step by step, but keep "
    "the reasoning compact. Place ONLY the final numeric or symbolic answer "
    "inside \\boxed{}: no units, no surrounding words, no trailing punctuation. "
    "If the problem has multiple [ANS] placeholders, output one final \\boxed{} "
    "with the sub-answers in the same order as the placeholders, comma "
    "separated, e.g. \\boxed{41, 35, 16}. Output the box exactly once, at the "
    "very end of your response."
)

SYSTEM_PROMPT_STRICT_MCQ = (
    "You are an expert mathematician. Read the problem and answer choices "
    "carefully, then choose the single best option. Output exactly one "
    "character inside \\boxed{}: the letter A-J of your choice, e.g. "
    "\\boxed{C}. Output the box exactly once, at the very end."
)

SYSTEM_PROMPT_COMMIT_NOW_MATH = (
    "You are an expert mathematician. State the answer immediately. Use at "
    "most a few lines of reasoning, then commit. End with exactly one "
    "\\boxed{...} containing only the final answer(s) in [ANS] placeholder "
    "order, comma separated for multi-part questions."
)

SYSTEM_PROMPT_COMMIT_NOW_MCQ = (
    "You are an expert mathematician. Choose the single best option. Be brief. "
    "End your response with exactly one \\boxed{X} where X is the letter A-J."
)

SYSTEM_PROMPT_REPAIR_BOX_MATH = (
    "You are a strict answer formatter. You will receive a math question and a "
    "draft model response. Extract or infer the final answer from the draft and "
    "output exactly one final \\boxed{...}. No extra prose. If there are "
    "multiple [ANS] placeholders, output one boxed group with comma-separated "
    "sub-answers in placeholder order."
)

SYSTEM_PROMPT_REPAIR_BOX_MCQ = (
    "You are a strict answer formatter. You will receive an MCQ question and a "
    "draft model response. Output exactly one final \\boxed{X} where X is a "
    "single letter A-J. No extra prose."
)

SYSTEM_PROMPT_BASELINE_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. If the problem has multiple "
    "sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_BASELINE_MCQ = (
    "You are an expert mathematician. Read the problem and the answer choices "
    "below, then select the single best answer. Output ONLY the letter of "
    "your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


SYSTEM_PROMPTS: dict[str, dict[str, str]] = {
    "strict": {
        "math": SYSTEM_PROMPT_STRICT_MATH,
        "mcq": SYSTEM_PROMPT_STRICT_MCQ,
    },
    "commit_now": {
        "math": SYSTEM_PROMPT_COMMIT_NOW_MATH,
        "mcq": SYSTEM_PROMPT_COMMIT_NOW_MCQ,
    },
    "baseline": {
        "math": SYSTEM_PROMPT_BASELINE_MATH,
        "mcq": SYSTEM_PROMPT_BASELINE_MCQ,
    },
    "repair_box": {
        "math": SYSTEM_PROMPT_REPAIR_BOX_MATH,
        "mcq": SYSTEM_PROMPT_REPAIR_BOX_MCQ,
    },
}


DEFAULT_PROMPT_ID = "strict"


def expected_num_answers(item: dict) -> int:
    """Number of answers expected for an item.

    MCQ items always expect exactly one boxed letter. Free-form items
    expect one boxed group containing one comma-separated value per
    ``[ANS]`` placeholder in the question; if the gold ``answer`` is a
    list we trust its length, otherwise fall back to counting
    ``[ANS]`` markers.
    """
    if item.get("options"):
        return 1
    gold = item.get("answer")
    if isinstance(gold, list):
        return max(1, len(gold))
    question = item.get("question", "")
    return max(1, question.count("[ANS]"))


def build_messages(
    item: dict,
    prompt_id: str = DEFAULT_PROMPT_ID,
) -> list[dict]:
    """Return Qwen chat messages [{role, content}, ...] for an item.

    The MCQ vs free-form distinction is determined by the presence of
    ``options`` in the JSONL row, matching the official starter notebook
    convention.
    """
    if prompt_id not in SYSTEM_PROMPTS:
        raise KeyError(
            f"unknown prompt_id={prompt_id!r}; available: "
            f"{sorted(SYSTEM_PROMPTS)}"
        )

    pack = SYSTEM_PROMPTS[prompt_id]
    question: str = item["question"]
    options: Optional[list] = item.get("options")

    if prompt_id == "repair_box":
        draft_response = item.get("_repair_draft", "")
        expected = expected_num_answers(item)
        if options:
            labels = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(
                f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options)
            )
            user_text = (
                f"Question:\n{question}\n\nOptions:\n{opts_text}\n\n"
                f"Draft response:\n{draft_response}\n\n"
                "Return only one final boxed letter, e.g. \\boxed{C}."
            )
            system_text = pack["mcq"]
        else:
            user_text = (
                f"Question:\n{question}\n\n"
                f"Expected number of sub-answers: {expected}\n\n"
                f"Draft response:\n{draft_response}\n\n"
                "Return only one final boxed answer, e.g. "
                "\\boxed{41, 35, 16}."
            )
            system_text = pack["math"]
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options)
        )
        user_text = f"{question}\n\nOptions:\n{opts_text}"
        system_text = pack["mcq"]
    else:
        user_text = question
        system_text = pack["math"]

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
