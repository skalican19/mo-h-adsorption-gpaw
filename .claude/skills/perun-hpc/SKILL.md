---
name: perun-hpc
description: Generic guide for writing and submitting SLURM batch scripts on the Perun HPC cluster (NSCC / HPC SAV). Covers access (ssh, keys), storage layout (/home /project /scratch /work), the environment (no scientific-software modules — bring your own; x86_64 CPU vs aarch64 GPU split), partitions & limits, accounting (--account, sprojects), copy-paste sbatch templates (single job, CPU array, GPU job), and monitoring. Use whenever writing an sbatch script for Perun, choosing a partition, setting up a cluster Python env, or transferring data to/from Perun.
---

# Running computations on Perun (NSCC / HPC SAV)

Generic guide for writing SLURM batch scripts on the **Perun** cluster.

**Source:** `https://userdocs.hpc.sav.sk/` (was `userdocs.nscc.sk` — 301-redirects to the new host).
**Verified 2026-07-21; aarch64 GPU env details confirmed on-cluster 2026-08-03.** Docs are a recently-migrated live site; re-check specifics on-cluster
(`sinfo`, `sprojects`, `quota -s`, `module avail`) before trusting a number here.

> **⚠️ Read the "Verify on-cluster" box at the bottom first.** Several quota numbers are
> unpublished, the accounting formula looks inherited from another cluster (Devana), and the
> partition table has internal inconsistencies. This guide flags each; don't treat unverified
> values as ground truth.

---

## 0. TL;DR

- **No common scientific-software modules** (no GPAW, VASP, Anaconda, etc.) — bring your own
  Python env. Site modules cover compilers, MPI, math libs, CUDA, and a few named apps (§3).
- **Architecture split is the #1 gotcha**: login + CPU compute nodes are x86_64; GPU nodes are
  aarch64 (Grace-Hopper GH200). An env built on one arch will NOT run on the other.
- **CUDA PyTorch on the GPU nodes needs an aarch64-specific wheel** — a plain `pip install torch`
  there is CPU-only and silently ignores the GPU. See §3.
- Keep code/small inputs on `/home` or `/project`; point large/temp outputs at
  `/scratch/<project_id>`; use `/work/$SLURM_JOB_ID` for hot per-job I/O.
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
| projects| `/project/<project_id>`      | **results/outputs**, shared project data | monthly (active only) | 6 mo after project end |
| scratch | `/scratch/<project_id>`       | temp job I/O, staging | **none** | **auto-purged** |
| work    | `/work/$SLURM_JOB_ID`         | node-local NVMe, hottest I/O | none | **deleted at job end** |

- Filesystems: Lustre for `/scratch` + `/project` (best at large sequential I/O), NFS for `/home`.
- **`/scratch` is not backed up and is purged** — copy results to `/project` before the job ends.
- No `$SCRATCH`/`$PROJECT` env var is documented — address scratch/project by `<project_id>` path.
- **Many small files hurt** the metadata servers → aggregate: `tar czf out.tar.gz outdir/`.

**Transfer (all via port 5522):**
```bash
scp -P 5522 -r <local> <user>@login.perun.sav.sk:<remote>
rsync -avhP -e "ssh -p 5522" <local>/ <user>@login.perun.sav.sk:<remote>/   # preferred for big trees
```
Check usage: `quota -s`, `du -sh /project/<id>/`, `sprojects -f`.

---

## 3. Environment — bring your own (arch-aware)

**There is no GPAW, VASP, or Anaconda module on Perun.** Site modules include Python 3.10–3.14,
GCC, Intel, OpenMPI/IntelMPI, FFTW, HDF5, MKL, CUDA, and (CPU nodes only) Quantum ESPRESSO 7.5 /
VASP 6.5.1 / SIESTA. Beyond that, you supply the Python env yourself.

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
  - CPU-only work → build the env on the **login node** (x86_64); it runs on `cn*`.
  - GPU/CUDA work → build the env **on a GPU node** (aarch64 wheels). This is the fiddly part —
    see the dedicated **"GPU env"** subsection below.

**Python env options**
- **venv:** `module load Python/<ver>` → `python3 -m venv <dir>` → `source <dir>/bin/activate`.
  In batch scripts, **load the same Python module** before sourcing the venv (version must match)
  — a venv built from a module Python needs that module loaded at runtime to find `libpython`.
- **conda:** install Miniconda manually (`Miniconda3-latest-Linux-x86_64.sh` for login/CPU,
  `-aarch64.sh` on a GPU node). One env per project; keep base minimal.
- If your Python build needed a rootless libffi/sqlite fallback (common on header-less nodes) and a
  batch/nohup shell strips library paths, export `LD_LIBRARY_PATH` to include those lib dirs before
  running Python.

### GPU env — CUDA / PyTorch on aarch64 (the hard part)

GPU work on Perun means running on the **aarch64 (ARM)** Grace-Hopper GH200 nodes (CUDA compute
capability **9.0**). The trap: **a plain `pip install torch` on aarch64 installs a CPU-only build
that silently ignores the GPU.** As of writing, the aarch64 CUDA wheel for torch 2.8 lives on the
**cu129** index (cu128 skips 2.8) — check `https://download.pytorch.org/whl/` for whatever the
current equivalent is when you hit this; a CPU-only env built on the login node will NOT produce a
CUDA torch here regardless.

Land on a GPU node first (interactive):
```bash
srun --partition=gpu_short --gres=gpu:1 --time=02:00:00 --pty bash    # aarch64 shell with a GPU
```

**Native aarch64 venv pattern:**
```bash
module load Python/3.12.3-GCCcore-13.3.0     # match whatever version your CUDA stack needs
python -m venv ~/envs/myenv && source ~/envs/myenv/bin/activate
pip install --upgrade pip
# torch FIRST, from the CUDA-enabled aarch64 index — installing other packages first can drag in
# a CPU-only torch off PyPI instead:
pip install --index-url https://download.pytorch.org/whl/cu129 "torch==2.8.0+cu129"
pip install <the rest of your GPU-dependent packages>
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect: ..., True
```
- The cu129 wheel bundles its own CUDA runtime — **no `CUDA/` module needed**; only the node driver
  must be new enough (fine on the GH200 nodes as of writing).
- A venv built from a module Python needs **that module loaded at runtime** to find `libpython`, so
  batch jobs must `module load` the same version before running the venv python.
- Packages with no prebuilt aarch64 wheels (some torch extension libraries, e.g. `torch_scatter` /
  `torch_sparse`) compile from source → `module load CUDA/<ver>` and
  `export TORCH_CUDA_ARCH_LIST=9.0` (Hopper) before installing.
- **Outbound internet:** GPU (aarch64) nodes **do have outbound internet** (confirmed — `pip
  install` and gated-model downloads both work on `gn*`). Build the venv once, then batch jobs can
  run fully offline afterward if your workload supports it (e.g. `HF_HUB_OFFLINE=1` for
  HuggingFace-gated models).
- **Containers:** a `.sif`/Singularity image built for x86_64 won't run here, and on at least one
  occasion a `.sif` failed to even mount on the GH200 kernel ("bad superblock … compression").
  Layering a GPU-python-stack container also risks the container's own CUDA/PyTorch build colliding
  with anything you `pip install` on top, via `LD_LIBRARY_PATH` ordering (`torch/lib` from the
  container winning over your own venv's). A native venv (above) sidesteps both problems — prefer
  it unless you have a specific reason to containerize.

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
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0   # or: source /path/to/your/venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python -u your_script.py --workers 1 --cores "${SLURM_CPUS_PER_TASK}"
```

### 7b. CPU array job — one task per work item
The general pattern for embarrassingly-parallel CPU work: build a manifest (one work-item id per
line), then:
```bash
#!/usr/bin/env bash
#SBATCH --job-name=cpu-array
#SBATCH --account=<project_id>
#SBATCH --partition=cpu_short          # cpu_long for jobs beyond the short-partition time limit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --array=1-100%20               # %20 = at most 20 tasks running at once
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
source /path/to/your/venv/bin/activate
ITEM="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${MANIFEST_PATH:?set MANIFEST_PATH via --export}")"
echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID}: ${ITEM} on $(hostname)"
python -u your_script.py --item "${ITEM}" --cores "${SLURM_CPUS_PER_TASK}"
```
Submit with the manifest path exported and `--array` sized to its line count:
```bash
N=$(wc -l < "$MANIFEST_PATH")
sbatch --array=1-"$N"%20 --export=ALL,MANIFEST_PATH="$MANIFEST_PATH" my_array.sh
```
`--array` forms: `1-8`, `1,3,9`, `1-7:2` (step), `1-100%10` (throttle). Cancel one/all:
`scancel <jobid>_[1-3]`, `scancel <jobid>`.

### 7c. GPU job
Requires the native aarch64 venv built in §3.
```bash
#!/usr/bin/env bash
#SBATCH --job-name=gpu-job
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
module load Python/3.12.3-GCCcore-13.3.0   # same module the venv was built from (finds libpython)
source ~/envs/myenv/bin/activate
python -u your_gpu_script.py --workers 1
```

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

## ⚠️ Verify on-cluster (docs gaps & inconsistencies)

- **Quotas for Perun `/home`, `/project`, `/scratch` are unpublished** ("content will be added").
  Get real numbers with `quota -s` and `sprojects -f`; don't assume another cluster's values.
- **BU/billing formula appears inherited from another cluster** (its doc example is headed "Devana
  nodes, 64 cores"). Confirm Perun charging before estimating core-hour cost.
- **Partition table inconsistencies:** `cpu_long` is listed with a 4-day limit but prose elsewhere
  says CPU max is 2 days; the "Memory (GB)" column doesn't match per-node RAM. Trust `sinfo` /
  `scontrol show partition` and the per-CPU MB figures over the table.
- **`--account` may or may not be strictly mandatory** (job-builder marks Project "optional", but
  examples elsewhere require it). Always set it; confirm with `sprojects`.
- **`sreport` is not documented**; use `sacct` / `seff` / `sstat` for usage & efficiency.
- **GPU/CUDA env is the fragile part, not SLURM** (§3): needs a CUDA-enabled *aarch64* PyTorch — the
  working wheel as of writing is `torch==2.8.0+cu129` (plain `pip install torch` is CPU-only). GPU
  nodes DO have outbound internet (confirmed). Check `torch.cuda.is_available()` before a real run.
- **Lmod's `module` is broken in non-login subshells on GPU nodes** (confirmed 2026-07-25): a
  `bash script.sh` / sbatch shell inherits a `module` function that resolves against the wrong-arch
  Lmod tree — `/apps/lmod` (x86) vs `/apps/lmod_gpu` (aarch64) — and fails, printing Lmod's Lua banner
  as shell errors (`… lmod: line N: -- : command not found`, `Copyright (C) …`). Fix: re-source the
  init matching the LIVE launcher, `source "$(dirname "$(dirname "$LMOD_CMD")")/init/bash"`, before
  `module load` (and wrap it in `set +u`) — derive the path from `$LMOD_CMD`, not `$LMOD_PKG`/
  `$MODULESHOME`, since those can point at the wrong-arch tree here.
- **`/home` is small-quota NFS** — keep big data (weights, outputs) on `/project`/`/scratch`, not `/home`.
- Docs recently moved `nscc.sk` → `hpc.sav.sk`; if a link 404s, swap the host.
