# CSE 151B Spring 2026 Math Reasoning Competition

This repo runs Qwen3-4B-Thinking-2507 against the unified-accuracy
benchmark. Inference happens inside a DSMLP GPU container via vLLM;
the local machine is only used for editing, validation harness output
inspection, and building the final submission CSV.

See `dsmlp/README.md` for the GPU workflow and `starter_code_cse151b_comp.ipynb`
for the original starter notebook.

## Pipeline at a glance

| Phase | Entry point | Where |
|-------|-------------|-------|
| Connect to GPU | `dsmlp/launch_gpu.sh` | dsmlp-login |
| Install env | `dsmlp/install_env.sh` | DSMLP pod |
| vLLM smoke test | `scripts/smoke_test_vllm.py --n 3` | DSMLP pod |
| Build val split | `scripts/build_validation_split.py` | anywhere |
| Run validation | `scripts/run_validation.py --run_id ...` | DSMLP pod |
| Build submission | `scripts/build_submission.py --test data/private.jsonl --out submission.csv` | DSMLP pod |
| (Optional) QLoRA | `scripts/train_qlora.py {build_data,train}` | DSMLP pod |

`scripts/run_validation.py` and `scripts/build_submission.py` both use
`src/qwen3_comp/self_consistency.py` so prompt selection, self-consistency
voting, retry policy, and token-budget routing are shared.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Original starter notebook (reference) |
| `src/qwen3_comp/` | Reusable inference modules (prompts, vLLM runtime, voting, scoring) |
| `scripts/` | CLI entry points for smoke test, validation, submission, QLoRA |
| `dsmlp/` | DSMLP GPU launch + install helpers and workflow README |
| `judger.py`, `utils.py` | Official scoring code (used by `qwen3_comp.scoring`) |
| `data/public.jsonl` | Public dataset (1126 questions) |
| `data/private.jsonl` | Private test set (943 questions; not in repo) |
| `results/` | Generated outputs: validation, submission, QLoRA artifacts |
| `submission.csv` | Final Kaggle upload (943 rows, `id,response` header) |

## Submission format

Kaggle expects exactly:

```
id,response
0,"Okay, let's try ... \boxed{42}"
1,"... \boxed{580, 660, 80}"
```

`scripts/build_submission.py` writes this with `pandas.to_csv`, then
reads it back to assert:

* row count matches the private set (default 943),
* header is exactly `id,response`,
* no duplicate or missing ids,
* no empty responses,
* boxed-answer rate ≥ 95% (warning otherwise).
