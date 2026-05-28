#!/usr/bin/env python3
"""Optional QLoRA SFT for Qwen3-4B-Thinking-2507.

This is the Phase 6 fine-tuning track from the plan. Run only inside a
DSMLP GPU container, and only after the inference-only baseline is
stable. The goal of this trainer is to teach the model to **commit a
final** ``\\boxed{...}`` **answer**, not to overfit on the public split.

Training data format
--------------------
A JSONL file where each row has ``messages``: a list of OpenAI-style
chat messages whose final ``assistant`` turn includes the gold final
``\\boxed{...}``. The split builder below emits this format directly
from ``data/public.jsonl`` minus a held-out validation split.

This script is intentionally minimal: LoRA on attention/MLP projections,
4-bit base model, short context, gradient checkpointing. It will not
beat the strict + self-consistency baseline by default. Use it only if
that baseline has plateaued.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen3_comp.data import load_jsonl, write_jsonl  # noqa: E402
from qwen3_comp.prompts import build_messages, expected_num_answers  # noqa: E402
from qwen3_comp.vllm_runtime import MODEL_ID, _require_cuda  # noqa: E402


def _gold_assistant_message(item: dict) -> str:
    """Construct the gold assistant turn for an SFT target.

    For multiple-choice we emit ``\\boxed{LETTER}``. For free-form we
    join the gold list into one comma-separated boxed group, matching
    the strict prompt's format.
    """
    gold = item.get("answer")
    if item.get("options"):
        return f"\\boxed{{{str(gold).strip().upper()}}}"
    if isinstance(gold, list):
        inner = ", ".join(str(g).strip() for g in gold)
    else:
        inner = str(gold).strip()
    return f"\\boxed{{{inner}}}"


def build_sft_jsonl(public_path: str, sft_out: str, val_out: str, seed: int = 42) -> None:
    """Convert ``data/public.jsonl`` into chat-formatted SFT rows.

    The held-out validation split (5% of items, stratified) is also
    saved so that fine-tuned candidates can be validated apples-to-apples
    against the strict baseline.
    """
    import random

    items = load_jsonl(public_path)
    rng = random.Random(seed)
    rng.shuffle(items)

    n_val = max(50, len(items) // 20)
    val_items = items[:n_val]
    train_items = items[n_val:]

    sft_rows: list[dict] = []
    for it in train_items:
        msgs = build_messages(it, prompt_id="strict")
        msgs.append({"role": "assistant", "content": _gold_assistant_message(it)})
        sft_rows.append(
            {
                "id": it.get("id"),
                "is_mcq": bool(it.get("options")),
                "expected_num": expected_num_answers(it),
                "messages": msgs,
            }
        )
    write_jsonl(sft_out, sft_rows)
    write_jsonl(val_out, val_items)
    print(f"Wrote SFT: {sft_out} ({len(sft_rows)} rows)")
    print(f"Wrote held-out validation: {val_out} ({len(val_items)} rows)")


def train(
    sft_path: str,
    out_dir: str,
    *,
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 16,
    max_seq_len: int = 4096,
) -> None:
    _require_cuda()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.gradient_checkpointing_enable()

    rows: list[dict] = []
    with open(sft_path) as f:
        for line in f:
            rec = json.loads(line)
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False
            )
            rows.append({"text": text})
    ds = Dataset.from_list(rows)

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved LoRA adapter to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    bd = sub.add_parser("build_data", help="emit SFT + heldout splits")
    bd.add_argument("--public", default="data/public.jsonl")
    bd.add_argument("--sft_out", default="results/qlora/sft.jsonl")
    bd.add_argument("--val_out", default="results/qlora/heldout.jsonl")
    bd.add_argument("--seed", type=int, default=42)

    tr = sub.add_parser("train", help="run QLoRA SFT on the SFT JSONL")
    tr.add_argument("--sft", default="results/qlora/sft.jsonl")
    tr.add_argument("--out", default="results/qlora/adapter")
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--batch_size", type=int, default=1)
    tr.add_argument("--grad_accum", type=int, default=16)
    tr.add_argument("--max_seq_len", type=int, default=4096)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "build_data":
        build_sft_jsonl(args.public, args.sft_out, args.val_out, seed=args.seed)
        return 0
    if args.cmd == "train":
        train(
            args.sft,
            args.out,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_seq_len=args.max_seq_len,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
