"""High-level helpers for the competition notebook and one-click workflows."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen3_comp.data import load_jsonl, stratify_split, summarise_split, write_jsonl  # noqa: E402
from qwen3_comp.extract import extract_boxed_content  # noqa: E402
from qwen3_comp.inference_config import FINAL_INFERENCE_DEFAULTS  # noqa: E402
from qwen3_comp.self_consistency import post_hoc_sanity  # noqa: E402


def setup_notebook() -> Path:
    """Ensure repo paths and vLLM-safe multiprocessing are configured."""
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return REPO_ROOT


def check_environment() -> dict[str, Any]:
    """Return CUDA/venv/path diagnostics for the first notebook cell."""
    import torch

    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    report = {
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "in_project_venv": str(REPO_ROOT / ".venv") in sys.executable,
        "venv_exists": venv_python.exists(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "private_exists": (REPO_ROOT / "data" / "private.jsonl").exists(),
        "public_exists": (REPO_ROOT / "data" / "public.jsonl").exists(),
    }
    return report


def install_environment() -> None:
    """Run the DSMLP bootstrap script (GPU pod only, one-time)."""
    script = REPO_ROOT / "dsmlp" / "bootstrap_venv.sh"
    if not script.exists():
        raise FileNotFoundError(script)
    subprocess.run(["bash", str(script)], cwd=REPO_ROOT, check=True)


@dataclass
class PipelineConfig:
    """Notebook convenience wrapper — defaults match ``inference_config.py``."""

    test_path: str = FINAL_INFERENCE_DEFAULTS["test_path"]
    out_csv: str = FINAL_INFERENCE_DEFAULTS["out_path"]
    responses_jsonl: str = FINAL_INFERENCE_DEFAULTS["responses_jsonl"]
    best_dir: str = "results/submission/best_0.636"
    rerun_checkpoint: str = "results/submission/responses_rerun_sane.jsonl"
    validation_split: str = "results/validation/sweep150.jsonl"
    expected_rows: int = FINAL_INFERENCE_DEFAULTS["expected_rows"]
    batch_size: int = FINAL_INFERENCE_DEFAULTS["batch_size"]
    n_mcq: int = FINAL_INFERENCE_DEFAULTS["n_mcq"]
    n_free: int = FINAL_INFERENCE_DEFAULTS["n_free"]
    primary_prompt: str = FINAL_INFERENCE_DEFAULTS["primary_prompt"]
    retry_prompt: str = FINAL_INFERENCE_DEFAULTS["retry_prompt"]
    enforce_eager: bool = FINAL_INFERENCE_DEFAULTS["enforce_eager"]
    max_num_seqs: int = FINAL_INFERENCE_DEFAULTS["max_num_seqs"]
    seed: int = FINAL_INFERENCE_DEFAULTS["seed"]
    recovery_primary_prompt: str = "commit_now"
    recovery_n_mcq: int = 7
    validation_n_mcq: int = 50
    validation_n_free_single: int = 80
    validation_n_free_multi: int = 20
    run_install: bool = False
    run_smoke_test: bool = False
    run_build_validation_split: bool = False
    run_validation: bool = False
    validation_run_id: str = "notebook_run"
    run_full_submission: bool = False
    run_prepare_recovery: bool = False
    run_targeted_recovery: bool = False
    rebuild_csv_only: bool = False
    skip_inference: bool = False


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_checkpoint_records(path: str | Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    p = _resolve(path)
    if not p.exists():
        return records
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def audit_checkpoint(path: str | Path, *, expected_rows: int = 943) -> dict[str, Any]:
    """Summarize boxed/sane coverage for a responses JSONL checkpoint."""
    p = _resolve(path)
    records = load_checkpoint_records(p)
    values = list(records.values())
    sanity_failures: Counter[str] = Counter()
    n_sane = 0
    n_boxed = 0
    n_truthy_boxed = 0

    for rec in values:
        resp = rec.get("response") or ""
        boxed = rec.get("boxed")
        if boxed is None:
            boxed = extract_boxed_content(resp)
        if boxed is not None:
            n_boxed += 1
        if boxed is not None and str(boxed).strip():
            n_truthy_boxed += 1
        if rec.get("sane") is not None:
            sane = bool(rec.get("sane"))
            reason = rec.get("sanity_reason", "ok")
        else:
            item = {
                "options": rec.get("options"),
                "question": rec.get("question", ""),
                "answer": rec.get("answer"),
            }
            sane, reason = post_hoc_sanity(item, resp)
        if sane:
            n_sane += 1
        else:
            sanity_failures[reason] += 1

    total = len(values)
    return {
        "path": str(p),
        "rows": total,
        "expected_rows": expected_rows,
        "missing_ids": max(0, expected_rows - total),
        "boxed_rate": n_boxed / max(1, total),
        "truthy_boxed_rate": n_truthy_boxed / max(1, total),
        "sane_rate": n_sane / max(1, total),
        "sanity_failures": dict(sanity_failures),
        "final_stage": dict(Counter(r.get("final_stage", "unknown") for r in values)),
        "repair_succeeded": sum(1 for r in values if r.get("repair_succeeded")),
    }


def freeze_best_submission(
    *,
    source_csv: str | Path = "submission.csv",
    source_checkpoint: str | Path = "responses.jsonl.backup",
    dest_dir: str | Path = "results/submission/best_0.636",
    kaggle_score: float = 0.636,
) -> dict[str, Any]:
    """Copy the current best submission artifacts to a safe directory."""
    dest = _resolve(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name, src in [
        ("submission.csv", source_csv),
        ("responses.jsonl.backup", source_checkpoint),
        ("submission.verification.json", "submission.verification.json"),
    ]:
        src_path = _resolve(src)
        if src_path.exists():
            shutil.copy2(src_path, dest / name)
            copied.append(name)
    manifest = {
        "kaggle_unified_accuracy": kaggle_score,
        "files": copied,
        "source_checkpoint": str(_resolve(source_checkpoint)),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if (dest / "responses.jsonl.backup").exists():
        audit = audit_checkpoint(dest / "responses.jsonl.backup")
        (dest / "audit.json").write_text(json.dumps(audit, indent=2))
        manifest["audit"] = audit
    return manifest


def build_validation_split(cfg: PipelineConfig) -> dict[str, Any]:
    items = load_jsonl(_resolve("data/public.jsonl"))
    subset = stratify_split(
        items,
        n_mcq=cfg.validation_n_mcq,
        n_free_single=cfg.validation_n_free_single,
        n_free_multi=cfg.validation_n_free_multi,
        seed=42,
    )
    out = _resolve(cfg.validation_split)
    write_jsonl(out, subset)
    summary = summarise_split(subset)
    meta = {
        "source": "data/public.jsonl",
        "out": str(out),
        "summary": summary,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def prepare_rerun_checkpoint(cfg: PipelineConfig) -> dict[str, Any]:
    from scripts.build_submission import _build_rerun_checkpoint

    src = _resolve(cfg.best_dir) / "responses.jsonl.backup"
    if not src.exists():
        src = _resolve("responses.jsonl.backup")
    dst = _resolve(cfg.rerun_checkpoint)
    report = _build_rerun_checkpoint(src, dst, mode="sane_boxed")
    report["audit"] = audit_checkpoint(dst)
    return report


def run_smoke_test(*, n: int = 3, backend: str = "vllm", seed: int = 42) -> int:
    from qwen3_comp.extract import extract_boxed_content
    from qwen3_comp.scoring import score_item
    from qwen3_comp.self_consistency import generate_with_retry_and_vote
    from qwen3_comp.vllm_runtime import VLLMEngine

    items = load_jsonl(_resolve("data/public.jsonl"))
    n_each = max(1, n // 3)
    subset = stratify_split(
        items,
        n_mcq=n_each,
        n_free_single=n_each,
        n_free_multi=max(0, n - 2 * n_each),
        seed=seed,
    )
    engine = VLLMEngine(backend=backend, seed=seed)
    results = generate_with_retry_and_vote(
        engine,
        subset,
        primary_prompt_id="strict",
        retry_prompt_id="commit_now",
        n_mcq=2,
        n_free=2,
        max_retries=1,
    )
    n_boxed = n_correct = 0
    rows = []
    for item, res in zip(subset, results):
        boxed = extract_boxed_content(res.response)
        ok = score_item(item, res.response)
        n_boxed += int(boxed is not None)
        n_correct += int(ok)
        rows.append(
            {
                "id": item.get("id"),
                "boxed": boxed,
                "correct": ok,
                "vote_answer": res.vote_answer,
            }
        )
    summary = {
        "n": len(results),
        "boxed": n_boxed,
        "correct": n_correct,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))
    return 0 if n_boxed > 0 else 1


def run_validation(cfg: PipelineConfig) -> dict[str, Any]:
    from scripts.run_validation import run_validation as _run_validation

    return _run_validation(
        split=str(_resolve(cfg.validation_split)),
        run_id=cfg.validation_run_id,
        n_mcq=cfg.n_mcq,
        n_free=cfg.n_free,
        primary_prompt=cfg.primary_prompt,
        retry_prompt=cfg.retry_prompt,
        seed=cfg.seed,
    )


def run_submission(cfg: PipelineConfig, **overrides: Any) -> dict[str, Any]:
    from scripts.build_submission import run_inference
    from qwen3_comp.inference_config import merge_inference_kwargs

    kwargs = merge_inference_kwargs(
        {
            "test_path": str(_resolve(cfg.test_path)),
            "out_path": str(_resolve(cfg.out_csv)),
            "responses_jsonl": str(_resolve(cfg.responses_jsonl)),
            "batch_size": cfg.batch_size,
            "n_mcq": cfg.n_mcq,
            "n_free": cfg.n_free,
            "primary_prompt": cfg.primary_prompt,
            "retry_prompt": cfg.retry_prompt,
            "enforce_eager": cfg.enforce_eager,
            "max_num_seqs": cfg.max_num_seqs,
            "seed": cfg.seed,
            "expected_rows": cfg.expected_rows,
            "skip_inference": cfg.skip_inference or cfg.rebuild_csv_only,
            "resume_require_sane_boxed": False,
        }
    )
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return run_inference(**kwargs)


def run_targeted_recovery(cfg: PipelineConfig) -> dict[str, Any]:
    prepare = prepare_rerun_checkpoint(cfg)
    report = run_submission(
        cfg,
        responses_jsonl=str(_resolve(cfg.rerun_checkpoint)),
        out_path=str(_resolve("results/submission/submission_recovery.csv")),
        primary_prompt=cfg.recovery_primary_prompt,
        retry_prompt=cfg.recovery_primary_prompt,
        n_mcq=cfg.recovery_n_mcq,
        resume_require_sane_boxed=True,
        skip_inference=False,
        rebuild_csv_only=False,
    )
    report["prepare"] = prepare
    report["post_audit"] = audit_checkpoint(_resolve(cfg.rerun_checkpoint))
    return report


def evaluate_qlora_gate(
    verification_path: str | Path,
    *,
    min_boxed_rate: float = 0.90,
    min_sane_rate: float = 0.88,
) -> dict[str, Any]:
    ver = json.loads(_resolve(verification_path).read_text())
    boxed = ver.get("boxed_rate", ver.get("truthy_boxed_rate", 0.0))
    sane = ver.get("sane_rate", 0.0)
    recommend = boxed >= min_boxed_rate and sane >= min_sane_rate
    report = {
        "recommend_qlora": recommend,
        "boxed_rate": boxed,
        "sane_rate": sane,
        "next_step": (
            "Optional QLoRA format SFT may help."
            if recommend
            else "Continue targeted recovery / prompt tuning; skip QLoRA for now."
        ),
    }
    out = _resolve("results/qlora/gate_decision.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def run_notebook_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    """Execute all steps enabled in *cfg* and return a summary dict."""
    setup_notebook()
    results: dict[str, Any] = {"environment": check_environment()}

    if cfg.run_install:
        install_environment()
        results["install"] = "ok"

    if cfg.run_build_validation_split:
        results["validation_split"] = build_validation_split(cfg)

    if cfg.run_smoke_test:
        results["smoke_test_exit"] = run_smoke_test()

    if cfg.run_validation:
        results["validation"] = run_validation(cfg)

    if cfg.run_prepare_recovery:
        results["prepare_recovery"] = prepare_rerun_checkpoint(cfg)

    if cfg.run_full_submission:
        results["submission"] = run_submission(cfg)

    if cfg.run_targeted_recovery:
        results["targeted_recovery"] = run_targeted_recovery(cfg)

    if cfg.rebuild_csv_only:
        cfg.skip_inference = True
        results["rebuild_csv"] = run_submission(cfg)

    checkpoint = _resolve(cfg.rerun_checkpoint if cfg.run_targeted_recovery else cfg.responses_jsonl)
    if checkpoint.exists():
        results["audit"] = audit_checkpoint(checkpoint)
        results["qlora_gate"] = evaluate_qlora_gate(checkpoint)

    return results
