# DSMLP GPU Workflow

**Gradescope verification:** run `python run_inference.py` inside the GPU pod.
See the root [`README.md`](../README.md) for full reproduction instructions.

Optional notebook: [`competition_pipeline.ipynb`](../competition_pipeline.ipynb)
(wraps the same `run_inference()` function).

All inference for the CSE 151B competition must run inside a DSMLP GPU
container. Do not run Qwen3-4B locally or on `dsmlp-login.ucsd.edu`.

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
cd ~/151B_SP26_Competition
bash dsmlp/install_env.sh
source .venv/bin/activate
```

This creates a project `.venv`, pins `numpy<2` (required for scipy in the
base image), installs `vllm==0.8.5.post1` and `transformers<5`, and prints
a CUDA capability check.

### Disk quota (`Disk quota exceeded`)

DSMLP home directories are small (~10–20 GB). The old bootstrap installed
`accelerate` before `vllm`, which downloaded **torch 2.12 + CUDA 13** (~3 GB)
and then **torch 2.6 + CUDA 12** again. If install fails mid-way:

```bash
bash dsmlp/free_disk.sh
git pull   # get fixed bootstrap_venv.sh
bash dsmlp/bootstrap_venv.sh
```

Do **not** `pip install accelerate` before `vllm` manually.

If `install_env.sh` prints `Installing pinned wheels` (old script) or you
see `numpy.core.multiarray` / `_ARRAY_API` errors, the pod has numpy 2.x in
`~/.local` and no `.venv`. **Sync the repo from your laptop first**, then:

```bash
cd ~/151B_SP26_Competition
bash dsmlp/bootstrap_venv.sh    # preferred one-shot fix
source .venv/bin/activate
python -c "import numpy, transformers; print(numpy.__version__, transformers.__version__)"
# expect: 1.26.x  4.xx  (NOT 2.x / 5.x)
```

From your laptop (replace `USERNAME`):

```bash
rsync -av --exclude .venv --exclude .git \
  ~/path/to/151B_SP26_Competition/ \
  USERNAME@dsmlp-login.ucsd.edu:~/151B_SP26_Competition/
```

If `.venv` already exists but imports fail:

```bash
bash dsmlp/fix_broken_env.sh
source .venv/bin/activate
```

**Always** activate `.venv` before `python scripts/...`. Running bare
`python` picks up broken packages from `~/.local`.

### Avoid Blackwell / MIG pods for vLLM

If `nvidia-smi` shows **MIG** or **Blackwell** (sm_120), vLLM's PyTorch 2.6
often will not run kernels. Delete the pod and relaunch requesting Ampere
or Ada class GPUs:

```bash
kubectl delete pod YOUR_POD_NAME
launch-scipy-ml.sh -g 1 -c 8 -m 32 -v a30
# or: -v b24gb
```

## 4. Run the vLLM smoke test

```bash
source .venv/bin/activate
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
source .venv/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_inference.py
```

`run_inference()` is the single Gradescope entry point: it loads the model,
runs the private set, applies all post-processing, and writes `submission.csv`.

## 6. Copy results back to your laptop

From your laptop:

```bash
rsync -av USERNAME@dsmlp-login.ucsd.edu:~/151B_SP26_Competition/results/ ./results/
rsync -av USERNAME@dsmlp-login.ucsd.edu:~/151B_SP26_Competition/submission.csv ./
```
