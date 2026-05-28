# DSMLP GPU Workflow

All inference for the CSE 151B competition must run inside a DSMLP GPU
container. The plan explicitly forbids running Qwen3-4B locally or on
`dsmlp-login.ucsd.edu`.

## 1. SSH to the submission host

From your laptop:

```bash
ssh USERNAME@dsmlp-login.ucsd.edu
```

Do not run Python/training directly on this host.

## 2. Launch a GPU container

From the SSH session:

```bash
cd ~/151B_SP26_Competition
bash dsmlp/launch_gpu.sh
```

The wrapper requests 1 GPU, 8 CPUs, 32 GB RAM, and 12 h pod timeout.

For long unattended runs use:

```bash
bash dsmlp/launch_gpu.sh -b                       # 6h background pod
bash dsmlp/launch_gpu.sh -B -- bash -lc 'python scripts/smoke_test_vllm.py'
```

Inside the container terminal verify GPU access first:

```bash
nvidia-smi
```

## 3. Install dependencies (inside the container)

```bash
bash dsmlp/install_env.sh
```

This installs `vllm==0.8.5.post1`, `transformers`, `accelerate`,
`bitsandbytes`, `pandas`, and `tqdm` and prints a CUDA visibility check.

## 4. Run the vLLM smoke test

```bash
python scripts/smoke_test_vllm.py --n 3
```

Expectations:

- 3 boxed responses out of 3.
- At least one correct, usually two.
- No `CUDA not available` errors.

The runtime module refuses to start unless CUDA is visible, so the
script will fail fast on the wrong host.

## 5. Run validation and build a submission

```bash
python scripts/run_validation.py --n_mcq 50 --n_free 100 --n_free_multi 20
python scripts/build_submission.py --test data/private.jsonl --out submission.csv
```

`build_submission.py` is the only script that should ever read
`data/private.jsonl`. It writes `id,response` for every test row,
includes the full reasoning trace per the Kaggle spec, and checks the
final file before exit.

## 6. Copy results back to your laptop

From your laptop:

```bash
rsync -av USERNAME@dsmlp-login.ucsd.edu:~/151B_SP26_Competition/results/ ./results/
rsync -av USERNAME@dsmlp-login.ucsd.edu:~/151B_SP26_Competition/submission.csv ./
```
