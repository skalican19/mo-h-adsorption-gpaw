"""
scripts/adsorbml/_common.py

Shared helpers for the AdsorbML pipeline (steps 1-3):
  - paths and env-configurable data root (ADSORBML_DATA_ROOT)
  - config constants (relaxation/screening settings, corrections)
  - logging setup, GPU detection
  - miller-index parsing, residual-force extraction
  - sharding for concurrent SLURM-array execution
  - atomic writes (crash-safe done-marker files)
  - the multi-GPU spawn/queue worker loop

Torch and fairchem are imported lazily (inside detect_gpus / the worker loop) so
that steps which do not touch a GPU (e.g. 3-extract_rank.py) stay lightweight.
"""
import ast
import logging
import multiprocessing as mp
import os
import re
from pathlib import Path

import numpy as np
from ase.io import read, write

# --- Paths / env-configurable data root -------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[2]
DATA_INPUTS = REPO_ROOT / "data" / "inputs" / "VASP_inputs"   # inputs stay in the repo

# Outputs may be redirected to fast scratch on HPC via ADSORBML_DATA_ROOT.
_ENV_ROOT   = os.environ.get("ADSORBML_DATA_ROOT")
OUTPUT_ROOT = Path(_ENV_ROOT) if _ENV_ROOT else REPO_ROOT / "data"

UMA_RELAXED  = OUTPUT_ROOT / "uma_relaxed"
MANIFEST_CSV = OUTPUT_ROOT / "adsorbml_manifest.csv"
OUT_DIR      = OUTPUT_ROOT / "adsorbml_results"

# --- Config constants --------------------------------------------------------
MIN_FREE_VRAM_GB   = 8.0
WORKERS_PER_GPU    = 1
FMAX               = 0.02        # eV/Å relaxation target (steps 1 & 2)
MAX_STEPS          = 100         # optimizer step cap
NUM_PLACEMENTS     = 100         # AdsorbML H* placements per slab (step 2)
ADSORBATE_SMILES   = "*H"
UMA_MODEL          = "uma-m-1p1"
ENTROPY_CORRECTION = 0.24        # eV, standard ZPE+entropy correction for H*
DEFAULT_EMIN       = -2.5        # eV, physical sanity window for E_ads (step 3)
DEFAULT_EMAX       = 2.5
FMAX_CONVERGED     = 0.05        # eV/Å, force-convergence threshold for diagnostics

LOG_FORMAT  = "%(asctime)s %(levelname)-8s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


# --- Logging -----------------------------------------------------------------
def setup_logging(log_path=None) -> None:
    """Configure the root logger once (idempotent). Optionally tee to a file."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    if log_path:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        root.addHandler(fh)


# --- GPU detection -----------------------------------------------------------
def detect_gpus(log=None) -> list:
    """Return eligible CUDA device ids (>= MIN_FREE_VRAM_GB free), most-free first.

    Respects SLURM's CUDA_VISIBLE_DEVICES (torch only sees allocated GPUs).
    Emits an inventory line per device to `log` (a logger) or to stdout.
    """
    import torch

    emit = log.info if log is not None else print
    if not torch.cuda.is_available():
        return []
    eligible = []
    for i in range(torch.cuda.device_count()):
        free_gb  = torch.cuda.mem_get_info(i)[0] / 1e9
        total_gb = torch.cuda.mem_get_info(i)[1] / 1e9
        ok = free_gb >= MIN_FREE_VRAM_GB
        status = "OK" if ok else f"LOW VRAM ({free_gb:.1f} GB free) – skipped"
        name = torch.cuda.get_device_properties(i).name
        emit(f"  GPU {i}: {name}  {free_gb:.1f}/{total_gb:.1f} GB  [{status}]")
        if ok:
            eligible.append((free_gb, i))
    eligible.sort(reverse=True)
    return [i for _, i in eligible]


# --- Miller indices ----------------------------------------------------------
def millers_from_name(name: str) -> tuple:
    """Extract miller indices from a structure name, e.g. MoS2_(110) -> (1,1,0)."""
    m = re.search(r'\((\d)(\d)(\d)\)', name)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (0, 0, 1)


def parse_millers(value) -> tuple:
    """Parse a stored miller value (tuple or its string repr) back to a tuple."""
    if isinstance(value, tuple):
        return value
    try:
        v = ast.literal_eval(str(value))
        return v if isinstance(v, tuple) else (0, 0, 1)
    except Exception:
        return (0, 0, 1)


# --- Residual force ----------------------------------------------------------
def fmax_from_traj(traj_path) -> float:
    """Max residual force (eV/Å) from a traj that stores forces; NaN otherwise."""
    if traj_path is None or traj_path == "" or (
        isinstance(traj_path, float) and np.isnan(traj_path)
    ):
        return float("nan")
    try:
        atoms = read(str(traj_path))
        if atoms.calc is None:
            return float("nan")
        forces = atoms.get_forces()
        return float(np.sqrt((forces ** 2).sum(axis=1)).max())
    except Exception:
        return float("nan")


# --- Sharding (concurrent SLURM-array safety) --------------------------------
def get_shard(cli_shard=None) -> tuple:
    """Return (i, n): this task's 0-based shard index and the total shard count.

    Priority: explicit --shard "I/N"  >  SLURM_ARRAY_TASK_ID/COUNT  >  (0, 1).
    SLURM ids are normalised by SLURM_ARRAY_TASK_MIN so any contiguous
    --array=start-end works. n == 1 means "process the whole list".
    """
    if cli_shard:
        i_str, n_str = str(cli_shard).split("/")
        i, n = int(i_str), int(n_str)
        if n < 1 or not (0 <= i < n):
            raise ValueError(f"Invalid --shard {cli_shard!r}: require 0 <= i < n and n >= 1")
        return i, n
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    cnt = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if tid is not None and cnt is not None:
        tmin = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
        return int(tid) - tmin, int(cnt)
    return 0, 1


def apply_shard(items, shard) -> list:
    """Return this shard's disjoint stride of `items` (whole list when n == 1)."""
    i, n = shard
    items = list(items)
    return items[i::n] if n > 1 else items


# --- Atomic writes (crash-safe) ----------------------------------------------
def write_atomic_csv(df, path) -> None:
    """Write a DataFrame to `path` via temp-file + os.replace (atomic rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_atomic_traj(atoms, path) -> None:
    """Write an ASE trajectory to `path` via temp-file + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    write(str(tmp), atoms, format="traj")
    os.replace(tmp, path)


# --- Multi-GPU worker loop ---------------------------------------------------
def _worker_loop(gpu_id, queue, task_name, process_item) -> None:
    """Worker entry point: build one UMA calculator, then drain the queue.

    Module-level so it is picklable under the 'spawn' start method. `process_item`
    must likewise be a module-level function in the calling script.
    """
    setup_logging()
    log = logging.getLogger("adsorbml.worker")
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cpu"

    from fairchem.core import FAIRChemCalculator
    log.info(f"Loading {UMA_MODEL} ({task_name}) on {device}...")
    calc = FAIRChemCalculator.from_model_checkpoint(UMA_MODEL, task_name=task_name, device=device)

    while True:
        item = queue.get()
        if item is None:
            break
        process_item(item, calc)


def run_gpu_workers(items, task_name, process_item, n_workers=None) -> None:
    """Spawn one worker per eligible GPU (or a single CPU worker) and drain `items`.

    Each worker builds FAIRChemCalculator(task_name) once, then calls
    process_item(item, calc) per queued item. Uses the 'spawn' context (required
    for CUDA); `process_item` must be a module-level (picklable) function.
    """
    if not items:
        return
    log = logging.getLogger("adsorbml.workers")
    gpu_ids = detect_gpus(log=log) or [None]
    n = n_workers or max(1, len(gpu_ids) * WORKERS_PER_GPU)
    log.info(f"Launching {n} worker(s) across device(s): {gpu_ids}")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for it in items:
        queue.put(it)
    for _ in range(n):
        queue.put(None)

    procs = [
        ctx.Process(target=_worker_loop,
                    args=(gpu_ids[k % len(gpu_ids)], queue, task_name, process_item))
        for k in range(n)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
