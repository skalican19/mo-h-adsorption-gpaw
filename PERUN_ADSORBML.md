# Running AdsorbML on Perun (GPU) — short guide

Runs AdsorbML steps **1 (relax)** and **2 (screen H\* sites)** on Perun's GPU nodes via an
NVIDIA NGC container. Step **3 (rank)** is CPU-only — run it locally. Three scripts:
`scripts/hpc_scripts/adsorbml/setup_perun_uma_env.sh`, `scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh`, `scripts/hpc_scripts/adsorbml/perun_adsorbml_worker.sh`.

## 0. Before you start (one-time)

- **Access:** `ssh -p 5522 <user>@login.perun.sav.sk` (note port 5522).
- **Project id:** run `sprojects` — you use it as `PROJECT_ID` / `ACCOUNT` below.
- **HuggingFace:** the `uma-m-1p1` model is *gated*. On huggingface.co: accept its license,
  then create an access token (Settings → Access Tokens). That token is your `hf_xxx`.
- **Get the repo onto Perun** (from the **login node** — it has internet; compute nodes may not):
  ```bash
  git clone <your-repo-url> ~/mo-h-adsorption-gpaw     # or `git pull` later to update
  ```
  If `git` is missing: `module load git` (find it with `module spider git`). Private-repo auth
  is either an HTTPS token or a *separate* SSH key added to GitHub (not your Perun login key).

## 1. Build the GPU env — once, on a GPU node

The env must be built on the aarch64 GPU arch, so grab a GPU node first:

```bash
srun --partition=gpu_short --gres=gpu:1 --time=02:00:00 --pty bash
cd ~/mo-h-adsorption-gpaw
PROJECT_ID=<proj> bash scripts/hpc_scripts/adsorbml/setup_perun_uma_env.sh --hf-token hf_xxx
```

This pulls the NGC PyTorch container, builds a fairchem venv, and downloads the model weights
to `/project/<proj>/hf_cache`. It should end by printing `torch.cuda.is_available() ... True`
and the `SIF_PATH` / `UMA_PYTHON` / `HF_HOME` values. Do this **once**; then `exit` the GPU node.

> If it can't reach the internet from the GPU node (pull or download fails), ask HPC support —
> you may need a proxy or a pre-staged image. Everything below depends on this step succeeding.

## 2. Run step 1 (relax) — from the login node

```bash
cd ~/mo-h-adsorption-gpaw
STEP=1 ACCOUNT=<proj> SIF_PATH=/project/<proj>/containers/pytorch-ngc.dir \
  INCLUDE="Mo2N_*" \
  bash scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh
```

- `INCLUDE` is optional (default = all structures); it scopes step 1 and, via the manifest,
  step 2 as well. Omit it to run everything.
- Prints a `sbatch` job id. Wait for it to finish before step 2.

## 3. Run step 2 (screen H\* sites) — after step 1 finishes

```bash
STEP=2 ACCOUNT=<proj> SIF_PATH=/project/<proj>/containers/pytorch-ngc.dir \
  bash scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh
```

(It refuses to submit until step 1's `adsorbml_manifest.csv` exists.)

## 4. Rank — locally (not on Perun)

```bash
python scripts/adsorbml/3-extract_rank.py     # -> data/adsorbml_results/ranked_candidates_stable.csv
```

## Monitor jobs

```bash
squeue -u "$USER"            # queue (R running, PD pending)
sacct -X -j <jobid>          # final state
seff <jobid>                 # CPU/GPU/mem efficiency (confirm the GPU was used)
scancel <jobid>              # cancel
```
Logs: `data/outputs/perun_logs/adsorbml-s<step>-perun_<jobid>.out|.err`.
Outputs: `data/uma_relaxed/`, `data/adsorbml_manifest.csv`, `data/adsorbml_results/`.

## Knobs (env vars on the submit command)

| Var | Default | Notes |
|-----|---------|-------|
| `STEP` | — (required) | `1` or `2` |
| `ACCOUNT` | — (required) | project id from `sprojects` |
| `SIF_PATH` | — (required) | container from step 1 (`/project/<proj>/containers/pytorch-ngc.dir`; a `--sandbox` dir, not a `.sif`) |
| `INCLUDE` | all | glob filter, step 1 only (e.g. `"Mo2N_*"`) |
| `GRES` | `gpu:1` | `gpu:4` = full node, ~4× faster (auto-parallel) |
| `PARTITION` | `gpu_short` | `gpu_medium` (2d) / `gpu_long` (4d) for longer runs |
| `TIME_LIMIT` | `12:00:00` | walltime |
| `ADSORBML_DATA_ROOT` | repo `data/` | set to `/scratch/<proj>` or `/project/<proj>` for big runs; **steps 1 & 2 must match** |
| `WORKERS` | auto (1/GPU) | override worker count |

## Gotchas

- **Set the right `NGC_TAG` in step 1.** The default is a guess — pick a real
  `nvcr.io/nvidia/pytorch` aarch64 tag (`NGC_TAG=<tag> bash scripts/hpc_scripts/adsorbml/setup_perun_uma_env.sh ...`).
- **Steps 1 and 2 must share `ADSORBML_DATA_ROOT`** (the manifest stores absolute paths).
- **`/scratch` is purged, not backed up** — if you stage there, copy results to `/projects`
  before the job's walltime ends.
- Full arch/partition/accounting detail lives in the `/perun-hpc` skill.
