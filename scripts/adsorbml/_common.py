"""
scripts/adsorbml/_common.py

Shared helpers for the AdsorbML pipeline (steps 1-3):
  - paths and env-configurable data root (ADSORBML_DATA_ROOT)
  - config constants (relaxation/screening settings, corrections)
  - logging setup, GPU detection
  - BestFrameLBFGS: the shared optimizer for steps 1 & 2 (keeps the lowest-fmax
    frame, records convergence) plus RELAX_INFO_KEYS / relax_outcome
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
from ase.optimize import LBFGS

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
# Step caps are split: step 1 does ~320 slab relaxations, so a generous cap is cheap
# and the slabs it produces are the input to everything downstream. Step 2 does
# ~320 x NUM_PLACEMENTS relaxations, where the cap dominates the GPU cost.
MAX_STEPS_SLAB      = 300        # optimizer step cap, step 1 (slab relaxation)
MAX_STEPS_PLACEMENT = 300        # optimizer step cap, step 2 (H* placements)
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
def atoms_fmax(atoms) -> float:
    """Max residual force (eV/Å) on the *constrained* forces — what the optimizer
    converges on. Assumes a calculator is attached; the value is cached by ASE, so
    calling this inside an optimizer observer costs no extra model evaluation."""
    forces = atoms.get_forces()
    return float(np.sqrt((forces ** 2).sum(axis=1)).max())


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


# --- Best-frame relaxation ---------------------------------------------------
# Keys BestFrameLBFGS stamps into atoms.info. Anything reading them must treat a
# missing key as UNKNOWN, never as converged (pre-2026-08 trajs have none of them).
RELAX_INFO_KEYS = (
    "relax_converged",    # bool  — forces reached relax_fmax_target
    "relax_nsteps",       # int   — optimizer steps actually taken
    "relax_max_steps",    # int   — the cap that was in force
    "relax_fmax_target",  # float — the fmax that was asked for
    "relax_fmax",         # float — fmax of the frame that was KEPT
    "relax_fmax_final",   # float — fmax of the LAST frame the optimizer visited
    "relax_best_step",    # int   — which step the kept frame came from
)


class BestFrameLBFGS(LBFGS):
    """LBFGS that keeps its lowest-fmax frame and records whether it converged.

    Two defects of plain ASE LBFGS motivate this:

    1. It has no line search, so forces/energy are not guaranteed to decrease. The
       final frame can be far worse than the initial one — in this project's data
       50/318 slabs ended worse than they started, the worst going from 24.6 to
       214.2 eV/Å. Whoever consumes the result has no way to tell.
    2. `run()` returns a convergence bool that callers routinely discard (including
       fairchem's own `relax_job`), and the step count is never recorded, so
       "converged" and "ran out of steps" are indistinguishable afterwards.

    On exit this restores the best positions **into the live Atoms object** and
    stamps RELAX_INFO_KEYS into `atoms.info`. Restoring in place is what makes this
    work inside fairchem's `relax_job`, which reads energy/forces from `atoms`
    *after* `run()` returns — those reads then describe the frame that was kept.
    Cost is one extra force evaluation per relaxation, and only when the best frame
    is not the last one.

    The observer fires once per force evaluation including step 0, so the kept frame
    is never worse than the input geometry.
    """

    def run(self, fmax=FMAX, steps=None):
        if steps is None:
            steps = MAX_STEPS_PLACEMENT
        atoms = self.atoms
        best = {"fmax": float("inf"), "step": -1, "positions": None}

        def _track_best():
            current = atoms_fmax(atoms)
            # NaN never wins this comparison, so exploded frames cannot be selected.
            if current < best["fmax"]:
                best.update(fmax=current, step=self.nsteps,
                            positions=atoms.get_positions())

        self.attach(_track_best, interval=1)
        converged = bool(super().run(fmax=fmax, steps=steps))
        final_fmax = atoms_fmax(atoms)

        # `not (best >= final)` rather than `best < final`, so a NaN final frame (forces
        # blew up) also loses to the best frame instead of silently winning.
        if best["positions"] is not None and not (best["fmax"] >= final_fmax):
            atoms.set_positions(best["positions"])
            kept_fmax, kept_step = best["fmax"], best["step"]
        else:
            kept_fmax, kept_step = final_fmax, self.nsteps

        atoms.info.update({
            "relax_converged":   converged,
            "relax_nsteps":      self.nsteps,
            "relax_max_steps":   steps,
            "relax_fmax_target": fmax,
            "relax_fmax":        kept_fmax,
            "relax_fmax_final":  final_fmax,
            "relax_best_step":   kept_step,
        })
        return converged


def relax_outcome(info) -> str:
    """Human-readable outcome from a RELAX_INFO_KEYS mapping (atoms.info or a CSV row)."""
    converged = info.get("relax_converged")
    nsteps    = info.get("relax_nsteps")
    cap       = info.get("relax_max_steps")
    if converged is None:
        return "UNKNOWN (no convergence flags recorded)"
    if converged:
        return "converged"
    if nsteps is not None and cap is not None and nsteps >= cap:
        return f"NOT CONVERGED (hit {cap}-step cap)"
    return f"NOT CONVERGED (optimizer stopped early at step {nsteps})"


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
