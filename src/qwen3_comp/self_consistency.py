"""Self-consistency voting and the strict/retry generation pipeline."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .budget import route_max_new_tokens
from .extract import (
    extract_boxed_content,
    extract_mcq_letter,
    is_valid_response,
    split_box_subanswers,
)
from .prompts import expected_num_answers
from .vllm_runtime import GenerationRequest, SamplingConfig, VLLMEngine


@dataclass
class VotingResult:
    """One question's vote outcome.

    ``response`` is the full text of the sample whose normalized answer
    won the vote; this is what gets written into the Kaggle CSV.
    """

    response: str
    vote_answer: Optional[str]
    vote_count: int
    n_samples: int
    extraction_rate: float
    samples: list[str]


_WS = re.compile(r"\s+")
_PURE_DECIMAL = re.compile(r"^-?\d+(\.\d+)?$")


def _float_canonical(s: str) -> Optional[str]:
    """Return a canonical decimal string for *s* if it parses as a pure float.

    Rounds to 6 significant figures so answers like ``442.857`` and
    ``442.857142857`` fall into the same vote bucket instead of splitting.
    Returns ``None`` when *s* is not a plain decimal (e.g. fractions or
    symbolic answers are left unchanged).
    """
    if not _PURE_DECIMAL.match(s):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if f == 0.0:
        return "0"
    # 6 significant figures → handles integer answers and common decimals
    formatted = f"{f:.6g}"
    # Strip trailing zeros after decimal point
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def normalize_answer(raw: str) -> str:
    """Normalise an answer string for majority-vote bucketing.

    Steps applied in order:
    1. Strip outer whitespace and LaTeX ``$`` delimiters.
    2. Collapse all internal whitespace (``\\frac{ a }{ b }`` → ``\\frac{a}{b}``).
    3. Unify fraction command variants.
    4. Strip ``\\left`` / ``\\right`` size hints.
    5. If the result looks like a plain decimal, round to 6 significant
       figures so near-equal values (e.g. ``442.857`` vs ``442.857142857``)
       map to the same bucket.
    """
    if raw is None:
        return ""
    out = raw.strip()
    out = out.strip("$ \t")
    out = _WS.sub("", out)
    out = out.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    out = out.replace("\\left", "").replace("\\right", "")
    canonical = _float_canonical(out)
    if canonical is not None:
        return canonical
    return out


def _vote_key(item: dict, response: str) -> Optional[str]:
    if item.get("options"):
        letter = extract_mcq_letter(response)
        return letter
    inner = extract_boxed_content(response)
    if inner is None:
        return None
    parts = split_box_subanswers(inner)
    if not parts:
        return None
    expected = expected_num_answers(item)
    if expected > 1 and len(parts) != expected:
        return None
    return "|".join(normalize_answer(p) for p in parts)


def majority_vote(item: dict, samples: list[str]) -> VotingResult:
    """Return the modal answer over a list of candidate completions."""
    extracted = 0
    counter: Counter[str] = Counter()
    key_to_sample: dict[str, str] = {}
    for resp in samples:
        key = _vote_key(item, resp)
        if key is None:
            continue
        extracted += 1
        counter[key] += 1
        key_to_sample.setdefault(key, resp)

    if not counter:
        return VotingResult(
            response=samples[0] if samples else "",
            vote_answer=None,
            vote_count=0,
            n_samples=len(samples),
            extraction_rate=0.0,
            samples=list(samples),
        )

    winner_key, votes = counter.most_common(1)[0]
    return VotingResult(
        response=key_to_sample[winner_key],
        vote_answer=winner_key,
        vote_count=votes,
        n_samples=len(samples),
        extraction_rate=extracted / max(1, len(samples)),
        samples=list(samples),
    )


def n_samples_for_item(item: dict, *, n_mcq: int, n_free: int) -> int:
    """Per-item self-consistency width.

    MCQ benefits more from voting because the label space is small;
    free-form voting is expensive and only helps when answers are
    cheap to normalise. The router defaults to higher ``n`` for MCQ
    and lower ``n`` for free-form.
    """
    return n_mcq if item.get("options") else n_free


def _make_request(
    item: dict,
    *,
    prompt_id: str,
    max_new_tokens: int,
    n: int,
    sampling: SamplingConfig,
) -> GenerationRequest:
    return GenerationRequest(
        item=item,
        prompt_id=prompt_id,
        max_new_tokens=max_new_tokens,
        n=n,
        sampling=sampling,
    )


def _repair_sampling(sampling: SamplingConfig) -> SamplingConfig:
    """Return conservative decoding settings for format repair."""
    return SamplingConfig(
        temperature=min(0.2, sampling.temperature),
        top_p=min(0.9, sampling.top_p),
        top_k=min(10, sampling.top_k),
        repetition_penalty=sampling.repetition_penalty,
        seed=sampling.seed,
    )


def _repair_max_new_tokens(item: dict) -> int:
    """Budget for the format-repair pass.

    Larger than the original 256/512 because the repair prompt sometimes
    needs a short reasoning trace to extract the correct sub-answer from
    a long, partially-truncated draft.
    """
    return 512 if item.get("options") else 1024


def generate_with_retry_and_vote(
    engine: VLLMEngine,
    items: list[dict],
    *,
    primary_prompt_id: str = "strict",
    retry_prompt_id: str = "commit_now",
    n_mcq: int = 5,
    n_free: int = 3,
    sampling: Optional[SamplingConfig] = None,
    max_retries: int = 1,
) -> list[VotingResult]:
    """Run the full strict + self-consistency + retry pipeline.

    The flow per question is:

    1. Generate ``n_mcq`` or ``n_free`` samples from the strict prompt
       at the type-routed token budget.
    2. Majority-vote on the boxed answers.
    3. If no sample produced an extractable boxed answer, retry once
       with the commit-now prompt at a larger token budget.
    """
    sampling = sampling or SamplingConfig()

    primary_requests: list[GenerationRequest] = []
    for it in items:
        primary_requests.append(
            _make_request(
                it,
                prompt_id=primary_prompt_id,
                max_new_tokens=route_max_new_tokens(it, retry=0),
                n=n_samples_for_item(it, n_mcq=n_mcq, n_free=n_free),
                sampling=sampling,
            )
        )

    primary_outputs = engine.generate(primary_requests)
    results: list[VotingResult] = []
    retry_indices: list[int] = []
    retry_requests: list[GenerationRequest] = []
    for idx, (item, samples) in enumerate(zip(items, primary_outputs)):
        vote = majority_vote(item, samples)
        results.append(vote)
        if vote.vote_answer is None and max_retries > 0:
            retry_indices.append(idx)
            retry_requests.append(
                _make_request(
                    item,
                    prompt_id=retry_prompt_id,
                    max_new_tokens=route_max_new_tokens(item, retry=1),
                    n=max(1, n_samples_for_item(item, n_mcq=n_mcq, n_free=n_free) // 2),
                    sampling=sampling,
                )
            )

    if retry_requests:
        retry_outputs = engine.generate(retry_requests)
        for idx, samples in zip(retry_indices, retry_outputs):
            item = items[idx]
            # Combine the new samples with the originals for voting
            combined = results[idx].samples + samples
            new_vote = majority_vote(item, combined)
            if new_vote.vote_answer is not None or not results[idx].samples:
                results[idx] = new_vote

    # Cheap repair pass: if we still have no valid vote key, ask the model to
    # emit only one final boxed answer from the existing draft response.
    repair_indices: list[int] = []
    repair_requests: list[GenerationRequest] = []
    repair_sampling = _repair_sampling(sampling)
    for idx, (item, vote) in enumerate(zip(items, results)):
        if vote.vote_answer is not None:
            continue
        draft = vote.response or (vote.samples[0] if vote.samples else "")
        if not draft:
            continue
        repair_item = dict(item)
        repair_item["_repair_draft"] = draft
        repair_indices.append(idx)
        repair_requests.append(
            _make_request(
                repair_item,
                prompt_id="repair_box",
                max_new_tokens=_repair_max_new_tokens(item),
                n=1,
                sampling=repair_sampling,
            )
        )

    if repair_requests:
        repair_outputs = engine.generate(repair_requests)
        for idx, samples in zip(repair_indices, repair_outputs):
            item = items[idx]
            combined = results[idx].samples + samples
            new_vote = majority_vote(item, combined)
            if new_vote.vote_answer is not None:
                results[idx] = new_vote

    # Final safety: each result must have a non-empty response field
    # so the submission CSV row is never empty.
    for idx, res in enumerate(results):
        if res.response:
            continue
        if res.samples:
            results[idx] = VotingResult(
                response=res.samples[0],
                vote_answer=None,
                vote_count=0,
                n_samples=res.n_samples,
                extraction_rate=res.extraction_rate,
                samples=res.samples,
            )

    return results


def post_hoc_sanity(item: dict, response: str) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for a final committed response."""
    return is_valid_response(
        response,
        is_mcq=bool(item.get("options")),
        expected_num=expected_num_answers(item),
    )
