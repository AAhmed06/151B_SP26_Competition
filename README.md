# CSE 151B Spring 2026 Math Reasoning Competition

Unified-accuracy submission using **Qwen/Qwen3-4B-Thinking-2507** (designated base model; no fine-tuning). All inference runs on a DSMLP GPU pod via vLLM. The full pipeline is exposed through a single function: `**run_inference()`** in `[run_inference.py](run_inference.py)`.

## GPU and inference time


| Item                      | Value                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **GPU**                   | NVIDIA **A30** (24 GB VRAM, Ampere) on UCSD DSMLP                                                                                      |
| **Launch**                | `bash dsmlp/launch_gpu.sh` (1 GPU, 8 CPU, 32 GB RAM)                                                                                   |
| **Approx. wall time**     | **3–4 hours** for all 943 private questions (`batch_size=64`, including primary generation, sanity-triggered retry, and repair passes) |
| **Engine**                | vLLM 0.8.5.post1, 4-bit bitsandbytes quantization, bfloat16                                                                            |
| **Reported Kaggle score** | Unified accuracy **~0.636** (stochastic; re-runs may vary slightly)                                                                    |


---

## Model weights setup

We did **not** fine-tune. Verification uses the designated HuggingFace base model only — no custom Hub upload is required.


| Setting             | Value                                                                     |
| ------------------- | ------------------------------------------------------------------------- |
| **Model ID**        | `Qwen/Qwen3-4B-Thinking-2507`                                             |
| **Download**        | Automatic on first inference (vLLM / HuggingFace Hub)                     |
| **Cache directory** | `~/.cache/huggingface/hub` (default Hub cache; no repo-local copy needed) |


---

## Reproduce results — `run_inference()`

### Environment (DSMLP GPU pod)

```bash
ssh USERNAME@dsmlp-login.ucsd.edu
cd ~/151B_SP26_Competition
bash dsmlp/launch_gpu.sh
bash dsmlp/bootstrap_venv.sh      # once per pod
source .venv/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Place `data/private.jsonl` (943 rows) in the repo. See `[dsmlp/README.md](dsmlp/README.md)` for troubleshooting.

### CLI (recommended)

```bash
python run_inference.py
```

### Python API

```python
from run_inference import run_inference

report = run_inference()
assert report["ok"]
```

**No other steps are required.** Calling `run_inference()` with default arguments produces the final `submission.csv`.

### Outputs


| File                                 | Description                                     |
| ------------------------------------ | ----------------------------------------------- |
| `submission.csv`                     | Final upload (`id,response`, 943 rows)          |
| `submission.verification.json`       | Structural checks (row count, boxed rate, etc.) |
| `results/submission/responses.jsonl` | Resumable checkpoint (re-run safely resumes)    |


---

## What `run_inference()` does (single entry point)

All logic is inside one call — nothing manual or external:

1. **Load model** — `Qwen/Qwen3-4B-Thinking-2507` via vLLM (`[src/qwen3_comp/vllm_runtime.py](src/qwen3_comp/vllm_runtime.py)`)
2. **Run inference** on `data/private.jsonl` with type-routed token budgets (`[src/qwen3_comp/budget.py](src/qwen3_comp/budget.py)`)
3. **Post-processing** (`[src/qwen3_comp/self_consistency.py](src/qwen3_comp/self_consistency.py)`):
  - Self-consistency majority vote (MCQ vs free-form sample widths)
  - Strict / commit-now prompts with sanity-triggered retry
  - `repair_box` formatter pass for failed extractions
  - Answer normalization and boxed extraction aligned with official `judger.py`
4. **Write** `submission.csv` and verification report

Implementation: `[scripts/build_submission.py](scripts/build_submission.py)` → `run_inference()`.

---

## Final hyperparameters

All defaults used for our submission are defined in `[src/qwen3_comp/inference_config.py](src/qwen3_comp/inference_config.py)`. Verification should call `run_inference()` **without overrides**.


| Parameter                         | Value                         |
| --------------------------------- | ----------------------------- |
| `model_id`                        | `Qwen/Qwen3-4B-Thinking-2507` |
| `backend`                         | `vllm`                        |
| `batch_size`                      | `64`                          |
| `n_mcq` / `n_free`                | `5` / `3`                     |
| `max_retries`                     | `1`                           |
| `primary_prompt` / `retry_prompt` | `strict` / `commit_now`       |
| `temperature` / `top_p` / `top_k` | `0.7` / `0.95` / `20`         |
| `seed`                            | `0`                           |
| `max_model_len`                   | `16384`                       |
| `max_num_seqs`                    | `96`                          |
| `gpu_memory_utilization`          | `0.75`                        |
| `enforce_eager`                   | `True`                        |


Sampling is stochastic (`temperature=0.7`). Outputs are not byte-identical across re-runs, but overall accuracy should stay consistent with our submission.

---

## Submission format

```
id,response
0,"Okay, let's try ... \boxed{42}"
1,"... \boxed{580, 660, 80}"
```

Each response includes the full model trace; the final answer must appear in `\boxed{...}`.

---

## Repository layout


| Path                                                                       | Purpose                                                                     |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `[run_inference.py](run_inference.py)`                                     | **Gradescope entry point** — `run_inference()`                              |
| `[src/qwen3_comp/inference_config.py](src/qwen3_comp/inference_config.py)` | Final hyperparameters                                                       |
| `[src/qwen3_comp/](src/qwen3_comp/)`                                       | Prompts, vLLM runtime, voting, scoring                                      |
| `[scripts/build_submission.py](scripts/build_submission.py)`               | `run_inference()` implementation                                            |
| `[data/public.jsonl](data/public.jsonl)`                                   | Public dev set (1126 questions, with answers)                               |
| `data/private.jsonl`                                                       | Private test set (943 questions; provide locally for verification)          |
| `[judger.py](judger.py)`, `[utils.py](utils.py)`                           | Official scoring utilities                                                  |
| `[competition_pipeline.ipynb](competition_pipeline.ipynb)`                 | Optional notebook wrapper (not required for verification)                   |
| `[dsmlp/](dsmlp/)`                                                         | DSMLP launch and install scripts — see `[dsmlp/README.md](dsmlp/README.md)` |


Optional development commands (smoke test, validation splits) are documented in `[dsmlp/README.md](dsmlp/README.md)`.