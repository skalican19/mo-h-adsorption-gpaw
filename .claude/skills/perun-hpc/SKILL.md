---
name: perun-hpc
description: How to write and submit SLURM batch scripts to run computations on the Perun HPC cluster (NSCC / HPC SAV). Covers access (ssh, keys), storage layout (/home /projects /scratch /work), the environment (no GPAW/VASP/conda modules — bring your own; x86_64 CPU vs aarch64 GPU split), partitions & limits, accounting (--account, sprojects), copy-paste sbatch templates (single CPU, CPU array for GPAW, GPU for fairchem/UMA), and monitoring. Use whenever writing an sbatch script for Perun, choosing a partition, setting up a cluster Python env, transferring data to/from Perun, or porting this repo's DEVANA launchers to Perun.
---

# Running computations on Perun (NSCC / HPC SAV)

Guide for writing SLURM batch scripts on the **Perun** cluster. Complements the repo's
existing DEVANA launchers (`scripts/submit_devana_gpaw_array.sh`,
`scripts/devana_gpaw_array_worker.sh`, `scripts/setup_pyenv_env.sh`) — Perun is a different,
newer cluster and needs different partitions and an architecture-aware environment.

**Source:** `https://userdocs.hpc.sav.sk/` (was `userdocs.nscc.sk` — 301-redirects to the new host).
**Verified 2026-07-21.** Docs are a recently-migrated live site; re-check specifics on-cluster
(`sinfo`, `sprojects`, `quota -s`, `module avail`) before trusting a number here.

> **⚠️ Read the "Verify on-cluster" box at the bottom first.** Several quota numbers are
> unpublished, the accounting formula looks inherited from Devana, and the partition table has
> internal inconsistencies. This guide flags each; don't treat unverified values as ground truth.

---

## 0. TL;DR for this repo

- **GPAW** (CPU-only) → `cpu_short` (≤1 day) or `cpu_long` (long jobs). One SLURM **array task
  per structure**, exactly like the DEVANA workflow. See the CPU-array template below.
- **fairchem / UMA** (Torch, GPU) → `gpu_short|gpu_medium|gpu_long`. **GPU nodes are ARM/aarch64 (GH200)**
  and need a **CUDA-enabled aarch64 PyTorch** — plain `pip install torch` is CPU-only and silently
  won't use the GPU. Build the env *on a GPU node* (NGC container or aarch64 CUDA wheels); see §3.
  It is a *separate* env from GPAW's (different arch).
- **No GPAW / VASP / Anaconda module exists on Perun** → bring your own env (reuse
  `scripts/setup_pyenv_env.sh`, which already builds libffi/sqlite rootless).
- Inputs stay in the repo (`/home` or `/projects`); point big outputs / `$ADSORBML_DATA_ROOT` at
  `/scratch/<project_id>` or `/projects/<project_id>`; use `/work/$SLURM_JOB_ID` for hot per-job I/O.
- **`--account=<project_id>` on every job.** Get the id from `sprojects -f`.

---

## 1. Access

```bash
ssh -p 5522 <user>@login.perun.sav.sk         # NOTE: port 5522, not 22
```
- Login nodes: `login01..login04.perun.sav.sk` (alias `login.perun.sav.sk`). No VPN for normal SSH.
- **Keys: ED25519 only.** `ssh-keygen -t ed25519 -o -a 100`, then upload the `.pub` at
  `https://register.hpc.sav.sk/sshkey`. Projects are managed at `register.hpc.sav.sk`.
- **Never run computations on login nodes.** Use `sbatch`, or `srun ... --pty bash` for interactive.

---

## 2. Storage — where things go

| Mount | Path | Use for | Backup | Persistence |
|-------|------|---------|--------|-------------|
| home    | `/home/<user>` (`$HOME`)      | code, scripts, configs, small data | daily | persistent |
| projects| `/projects/<project_id>`      | **results/outputs**, shared project data | monthly (active only) | 6 mo after project end |
| scratch | `/scratch/<project_id>`       | temp job I/O, staging | **none** | **auto-purged** |
| work    | `/work/$SLURM_JOB_ID`         | node-local NVMe, hottest I/O | none | **deleted at job end** |

- Filesystems: Lustre for `/scratch` + `/projects` (best at large sequential I/O), NFS for `/home`.
- **`/scratch` is not backed up and is purged** — copy results to `/projects` before the job ends.
- No `$SCRATCH`/`$PROJECT` env var is documented — address scratch/projects by `<project_id>` path.
- **Many small files hurt** the metadata servers → aggregate: `tar czf out.tar.gz outdir/`.

**Transfer (all via port 5522):**
```bash
scp -P 5522 -r <local> <user>@login.perun.sav.sk:<remote>
rsync -avhP -e "ssh -p 5522" <local>/ <user>@login.perun.sav.sk:<remote>/   # preferred for big trees
```
Check usage: `quota -s`, `du -sh /projects/<id>/`, `sprojects -f`.

---

## 3. Environment — bring your own (arch-aware)

**There is no GPAW, VASP, or Anaconda module on Perun.** Site modules include Python 3.10–3.14,
GCC, Intel, OpenMPI/IntelMPI, FFTW, HDF5, MKL, CUDA, and (CPU nodes only) Quantum ESPRESSO 7.5 /
VASP 6.5.1 / SIESTA. For GPAW + fairchem you supply the Python env yourself.

**Lmod:**
```bash
module avail            # what's installable
module spider <name>    # find a module + how to load it
module load Python/3.12.3-GCCcore-13.3.0
module list ; module purge      # ml = alias for module
```

### ⚠️ Architecture split (the #1 Perun gotcha)
- **Login + CPU compute nodes = x86_64** (AMD EPYC Turin).
- **GPU nodes = aarch64 / ARM** (Grace-Hopper GH200).
- **A venv/conda env built on the login node will NOT run on the GPU nodes**, and vice versa.
  - GPAW / CPU work → build the env on the **login node** (x86_64); it runs on `cn*`.
  - fairchem / UMA / Torch → build the env **on a GPU node** (aarch64 wheels). This is the fiddly
    part — see the dedicated **"GPU env for fairchem / UMA"** subsection below.

**Python env options**
- **Reuse the repo bootstrap** (recommended for GPAW): `bash scripts/setup_pyenv_env.sh` — it builds
  a pyenv Python with a rootless libffi/sqlite fallback (handles header-less clusters) and installs
  `requirements.txt`. Run it once on the login node for the CPU env.
- **venv:** `module load Python/<ver>` → `python3 -m venv <dir>` → `source <dir>/bin/activate`.
  In batch scripts, **load the same Python module** before sourcing the venv (version must match).
- **conda:** install Miniconda manually (`Miniconda3-latest-Linux-x86_64.sh` for login/CPU,
  `-aarch64.sh` on a GPU node). One env per project; keep base minimal.
- If a batch/nohup shell strips library paths, export before running Python (see `setup_pyenv_env.sh`):
  `export LD_LIBRARY_PATH="$HOME/.local/mo_h_bootstrap/lib64:$HOME/.local/mo_h_bootstrap/lib:$LD_LIBRARY_PATH"`

### GPU env for fairchem / UMA — the hard part (aarch64 + CUDA PyTorch)

GPU screening on Perun means running Torch/fairchem on the **aarch64 (ARM)** Grace-Hopper GH200
nodes (CUDA compute capability **9.0**). The trap: **a plain `pip install torch` on aarch64 installs
a CPU-only build that silently ignores the GPU** — you must obtain a CUDA-enabled *aarch64* PyTorch.
Build the env **on a GPU node** so wheels match the arch; `setup_pyenv_env.sh` is for the CPU/GPAW
env and will NOT produce a CUDA Torch here.

Land on a GPU node first (interactive):
```bash
srun --partition=gpu_short --gres=gpu:1 --time=02:00:00 --pty bash    # aarch64 shell with a GPU
```

**Option A — NVIDIA NGC PyTorch container (recommended, most robust on GH200):**
```bash
module load singularity
# Pull an aarch64 image (needs an aarch64 host WITH internet — see the caveat below). Pick a recent tag.
singularity pull docker://nvcr.io/nvidia/pytorch:25.xx-py3
# Put fairchem in a venv that reuses the container's CUDA Torch (do not reinstall torch):
singularity exec --nv pytorch_25.xx-py3.sif python -m venv --system-site-packages ~/envs/uma
singularity exec --nv pytorch_25.xx-py3.sif ~/envs/uma/bin/pip install fairchem-core fairchem-data-oc
# Verify the GPU is visible from inside the container:
singularity exec --nv pytorch_25.xx-py3.sif ~/envs/uma/bin/python \
  -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect True
```

**Option B — native aarch64 pip env (no container):**
```bash
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.8.0
python -m venv ~/envs/uma-aarch64 && source ~/envs/uma-aarch64/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # aarch64 CUDA wheel (torch>=2.7)
pip install fairchem-core fairchem-data-oc
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # MUST print True
```
- Match the `cuXXX` wheel index to the loaded `CUDA/` module (cu128 ↔ CUDA 12.8). torch ≥2.11 also
  ships aarch64 CUDA wheels on plain PyPI (no `--index-url` needed).
- **`torch_scatter` / `torch_sparse`** (if a fairchem dep pulls them) have no prebuilt aarch64 wheels →
  they compile from source: keep the `CUDA/` module loaded and set `export TORCH_CUDA_ARCH_LIST=9.0`
  (Hopper) before installing.

**⚠️ Operational unknown — verify on-cluster:** both options need outbound internet *on an aarch64
host* to fetch wheels/images. Login nodes have internet but are **x86_64** (wrong arch); GPU nodes are
aarch64 but may lack outbound access. If GPU nodes have no internet, ask HPC support about a proxy or a
pre-staged image/wheel cache. Build the `.sif`/venv **once**, then just reuse it in batch jobs.

---

## 4. Partitions & limits

| Partition | Nodes | Max time | Max nodes/cores | GPUs | Notes |
|-----------|-------|----------|-----------------|------|-------|
| `testing`      | login01–04 | 0-00:30 | 1 / 160 | 0 | short validation only |
| `cpu_short` *  | cn001–045  | 1-00:00 | 2 / 640 | 0 | **default** |
| `cpu_long`     | cn001–045  | (≤4-00:00 — verify) | 1 / 320 | 0 | long CPU jobs |
| `cpu_hm_short` | cn046–060  | 1-00:00 | 1 / 320 | 0 | high-mem, 1 node/job |
| `cpu_hm_long`  | cn046–060  | 4-00:00 | 1 / 320 | 0 | high-mem, 1 node/job |
| `gpu_short`    | gn001–076  | 1-00:00 | 4 / 1152 | 4/node | GH200, **aarch64** |
| `gpu_medium`   | gn001–076  | 2-00:00 | 2 / 576 | 4/node | GH200, aarch64 |
| `gpu_long`     | gn001–076  | 4-00:00 | 1 / 288 | 4/node | GH200, aarch64 |

Time format is `D-HH:MM`. Select with `-p <name>` / `--partition=<name>`.

**Node hardware**
- **CPU (cn001–045, universal):** 2× AMD EPYC Turin 9845 = **320 cores/node**, 1152 GB DDR5, 3.8 TB local NVMe.
- **CPU high-mem (cn046–060):** same cores, **2304 GB** RAM, 7.6 TB NVMe.
- **GPU (gn001–076):** 4× Grace-Hopper GH200, **288 Arm cores/node**, 96 GB HBM3 per GPU, 3.8 TB NVMe.

**Memory guidance:** request via `--mem-per-cpu` using **≈3450 MB/CPU** on standard nodes,
**≈6900 MB/CPU** on high-mem nodes. (The docs' "Memory (GB)" partition column is ambiguous — see caveats.)

Inspect live: `sinfo`, `sinfo -p <part>`, `scontrol show partition <part>`.

---

## 5. Accounting & priority

- **`--account=<project_id>`** charges the job to your project. Find it / check allocation:
  ```bash
  sprojects        # your projects
  sprojects -a     # allocations as spent/awarded (e.g. CPU: 10/50000)
  sprojects -f     # full info incl. shared storage paths + members
  ```
- **Billing (BU)** — *unverified for Perun, flag before trusting:*
  `CPU: BU = MAX(cores, GB*0.256)`; `GPU: BU = MAX(cores, GB*0.256, GPUs*16)`.
  Out of allocation → `squeue` reason `(QOSGrpBillingMinutes)`.
- **Priority / fairshare:** based on usage over the last 14 days; check with
  `sprio -S -y`, `sshare -A <project>`, `squeue --start -j <jobid>`.

---

## 6. SBATCH script anatomy

Directives go at the top as `#SBATCH <flag>`. Short and long forms both work:

| Purpose | Short | Long |
|---------|-------|------|
| job name | `-J` | `--job-name=` |
| project (charge) | `-A` | `--account=` |
| partition | `-p` | `--partition=` |
| nodes | `-N` | `--nodes=` |
| tasks (total) | `-n` | `--ntasks=` |
| tasks/node | — | `--ntasks-per-node=` |
| CPUs/task | `-c` | `--cpus-per-task=` |
| mem/CPU | — | `--mem-per-cpu=` (or `--mem=` per node) |
| walltime | `-t` | `--time=` (`D-HH:MM:SS` or `HH:MM:SS`) |
| stdout / stderr | `-o` / `-e` | `--output=` / `--error=` (`%j` job id, `%A_%a` array) |
| array | `-a` | `--array=` |
| GPUs | `-G` | `--gpus=` (or `--gres=gpu:N`, `--mem-per-gpu=`) |
| email | — | `--mail-user=` `--mail-type=begin,end,fail` |

Submit / interactive:
```bash
sbatch myjob.sh
srun --partition=cpu_short --nodes=1 --ntasks=8 --time=02:00:00 --pty bash   # interactive shell
```

---

## 7. Copy-paste templates

### 7a. Single CPU job
```bash
#!/usr/bin/env bash
#SBATCH --job-name=serial
#SBATCH --account=<project_id>
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11          # matches repo CORES_PER_CALC
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
source "$HOME/.pyenv/versions/cemea-env/bin/activate"   # or: module load Python/... ; source venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python -u scripts/gpaw_h_adsorption.py --structure-name Mo2N_slab --workers 1 \
       --cores-per-calc "${SLURM_CPUS_PER_TASK}"
```

### 7b. CPU array — one task per structure (GPAW; mirrors the DEVANA workflow)
This is the primary pattern. Build a manifest (one structure name per line) exactly as
`submit_devana_gpaw_array.sh` does via `gpaw_h_adsorption.py --write-structure-list`, then:
```bash
#!/usr/bin/env bash
#SBATCH --job-name=gpaw-perun
#SBATCH --account=<project_id>
#SBATCH --partition=cpu_short          # cpu_long for jobs > 1 day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=11             # CORES_PER_CALC
#SBATCH --mem-per-cpu=4G               # ≈ RAM_PER_CALC_GB
#SBATCH --time=24:00:00
#SBATCH --array=1-350%20               # %20 = at most 20 tasks running at once
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
source "$HOME/.pyenv/versions/cemea-env/bin/activate"
# resolve this task's structure from the manifest (1 line = 1 structure)
STRUCTURE_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST_PATH:?set MANIFEST_PATH via --export}")"
echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID}: ${STRUCTURE_NAME} on $(hostname)"
python -u scripts/gpaw_h_adsorption.py \
    --structure-name "${STRUCTURE_NAME}" --workers 1 \
    --cores-per-calc "${SLURM_CPUS_PER_TASK}" --relax-steps 200 --fmax 0.03
```
Submit with the manifest path exported and `--array` sized to its line count:
```bash
N=$(wc -l < "$MANIFEST_PATH")
sbatch --array=1-"$N"%20 --export=ALL,MANIFEST_PATH="$MANIFEST_PATH" gpaw_array.sh
```
`--array` forms: `1-8`, `1,3,9`, `1-7:2` (step), `1-100%10` (throttle). Cancel one/all:
`scancel <jobid>_[1-3]`, `scancel <jobid>`.

### 7c. GPU job — fairchem / UMA screening (uses the aarch64 env from §3)
Requires the CUDA-enabled aarch64 env built in §3 (container `.sif` **or** native venv). fairchem
also uses the GPU via `--nv`; GPAW re-checks the ranked candidates afterward on the CPU partitions (§7b).
```bash
#!/usr/bin/env bash
#SBATCH --job-name=uma-relax
#SBATCH --account=<project_id>
#SBATCH --partition=gpu_short          # gpu_medium/long for longer runs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   # 4 GPUs/node; --gres=gpu:4 = a full node
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- Option A: NGC container (recommended) ---
module load singularity
singularity exec --nv -B /scratch,/projects "$HOME/pytorch_25.xx-py3.sif" \
    "$HOME/envs/uma/bin/python" -u scripts/adsorbml/1-relax_uma_omat.py --include "Mo2N_*" --workers 1

# --- Option B: native venv (use EITHER A or B, not both) ---
# module load CUDA/12.8.0
# source "$HOME/envs/uma-aarch64/bin/activate"
# python -u scripts/adsorbml/1-relax_uma_omat.py --include "Mo2N_*" --workers 1
```
The rest of the ML pipeline — `2-run_adsorbml.py` (screen H* sites) and `3-extract_rank.py` (rank) —
runs the same way on GPU; both steps also support SLURM-array sharding (`--shard I/N`, or auto from
`SLURM_ARRAY_TASK_ID`/`COUNT`) if you want to fan the screening across many GPU tasks.

---

## 8. Monitor & manage

```bash
squeue -u "$USER"                 # my queue (ST col: PD pending, R running, CG completing)
sacct -X -j <jobid>               # one job, history          (F failed, TO timeout, OOM oom)
seff <jobid>                      # CPU/mem/GPU efficiency after it finishes
sstat --jobs=<jobid> --format JobID,MaxRSS,AveRSS   # live stats of a running job
scancel <jobid>                   # cancel;  scancel <jobid>_[1-3] for array elements
scontrol show job <jobid>         # full detail / why pending
```
Use `seff` to right-size `--cpus-per-task` / `--mem-per-cpu` for the next batch (avoid over-billing:
memory above ~3450 MB/CPU pushes the BU up via the `GB*0.256` term).

---

## 9. Porting this repo's DEVANA scripts to Perun

The DEVANA launchers already encode the right pattern (manifest → array → worker → pyenv env).
To target Perun, the deltas are:
- **Partition:** DEVANA `PARTITION=cpu` → Perun `cpu_short` (default) or `cpu_long`. GPU work →
  `gpu_short|gpu_medium|gpu_long`.
- **Env:** run `scripts/setup_pyenv_env.sh` once on the **login node** (x86_64) for the GPAW/CPU env;
  build any fairchem/UMA env **on a GPU node** (aarch64). Keep the `LD_LIBRARY_PATH` bootstrap export.
- **Walltime:** cap per partition (`cpu_short` ≤ 1 day; use `cpu_long` beyond that — verify the max).
- **Storage:** set `$ADSORBML_DATA_ROOT` to `/scratch/<id>` (staging) or `/projects/<id>` (keep),
  and copy finals off `/scratch` before job end. Inputs stay in the repo.
- Everything else (`--account`, `--array=1-N`, `SLURM_ARRAY_TASK_ID`→structure, `--export=ALL,...`)
  carries over unchanged.

*(No Perun launcher scripts exist yet — writing `submit_perun_gpaw_array.sh` + worker is a separate
task. Use the templates in §7 until then.)*

---

## ⚠️ Verify on-cluster (docs gaps & inconsistencies)

- **Quotas for Perun `/home`, `/projects`, `/scratch` are unpublished** ("content will be added").
  Get real numbers with `quota -s` and `sprojects -f`; don't assume the Devana values.
- **BU/billing formula appears inherited from Devana** (its doc example is headed "Devana nodes, 64
  cores"). Confirm Perun charging before estimating core-hour cost.
- **Partition table inconsistencies:** `cpu_long` is listed with a 4-day limit but prose elsewhere
  says CPU max is 2 days; the "Memory (GB)" column doesn't match per-node RAM. Trust `sinfo` /
  `scontrol show partition` and the per-CPU MB figures over the table.
- **`--account` may or may not be strictly mandatory** (job-builder marks Project "optional", but
  examples and the repo's DEVANA submitter require it). Always set it; confirm with `sprojects`.
- **`sreport` is not documented**; use `sacct` / `seff` / `sstat` for usage & efficiency.
- **GPU/fairchem env is the fragile part, not SLURM** (§3): needs a CUDA-enabled *aarch64* PyTorch —
  plain `pip install torch` is CPU-only; use the cu128 aarch64 wheel or an NGC container, and possibly
  a `torch_scatter` source build (`TORCH_CUDA_ARCH_LIST=9.0`). Also confirm GPU nodes have outbound
  internet for the install, and check `torch.cuda.is_available()` before launching a real run.
- Docs recently moved `nscc.sk` → `hpc.sav.sk`; if a link 404s, swap the host.
