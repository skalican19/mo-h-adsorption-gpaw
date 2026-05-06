"""
scripts/adsorbml/1-relax_uma_omat.py

AdsorbML step 1: Relax POSCAR structures with UMA-M (OMAT head) and write
a manifest CSV for the next step.

Excluded: structures with 'graphene', 'nanoribbon', or 'edge' in their name.

Outputs:
  data/uma_relaxed/<name>.traj   — relaxed slab per structure
  data/adsorbml_manifest.csv     — manifest for 2-run_adsorbml.py

Usage:
  python scripts/adsorbml/1-relax_uma_omat.py
  python scripts/adsorbml/1-relax_uma_omat.py --include "MoS2_*,Mo2N_*"
  python scripts/adsorbml/1-relax_uma_omat.py --workers 2
"""
import sys
import os
import argparse
import logging
import multiprocessing as mp
import re
from pathlib import Path

import pandas as pd
import torch
from ase.io import read, write
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator

# Import discover_structures from the sibling scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpaw_h_adsorption import discover_structures

REPO_ROOT       = Path(__file__).resolve().parents[2]
DATA_INPUTS     = REPO_ROOT / "data" / "inputs" / "VASP_inputs"
UMA_RELAXED     = REPO_ROOT / "data" / "uma_relaxed"
MANIFEST_CSV    = REPO_ROOT / "data" / "adsorbml_manifest.csv"

MIN_FREE_VRAM_GB = 8.0
WORKERS_PER_GPU  = 1
FMAX             = 0.02
MAX_STEPS        = 200

# Structures excluded from AdsorbML (not 2D-periodic surface slabs or off-topic)
_EXCLUDE = ("graphene", "nanoribbon", "edge")


def _is_excluded(name: str) -> bool:
    low = name.lower()
    return any(pat in low for pat in _EXCLUDE)


def _parse_millers(name: str) -> tuple:
    """Extract miller indices tuple from structure name, e.g. MoS2_(110) → (1,1,0)."""
    m = re.search(r'\((\d)(\d)(\d)\)', name)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (0, 0, 1)


def _tag_atoms(atoms):
    """Assign surface tags needed by AdsorbML: 1=surface, 2=subsurface, 0=bulk."""
    z_max = atoms.positions[:, 2].max()
    tags = []
    for atom in atoms:
        z = atom.position[2]
        if z > z_max - 1.5:
            tags.append(1)
        elif z > z_max - 4.0:
            tags.append(2)
        else:
            tags.append(0)
    atoms.set_tags(tags)
    return atoms


def _relax_one(name: str, poscar_path: Path, calc) -> None:
    log = logging.getLogger(f"relax.{name}")
    out_traj = UMA_RELAXED / f"{name}.traj"

    if out_traj.exists():
        log.info(f"SKIP (already done): {name}")
        return

    log.info(f"Start: {name}")
    try:
        atoms = read(str(poscar_path))
        atoms = _tag_atoms(atoms)
        atoms.calc = calc
        opt = LBFGS(atoms, logfile=str(UMA_RELAXED / f"{name}_opt.log"))
        opt.run(fmax=FMAX, steps=MAX_STEPS)
        write(str(out_traj), atoms)
        log.info(f"Done: {name}  E={atoms.get_potential_energy():.4f} eV")
    except Exception as exc:
        log.error(f"Failed {name}: {exc}")


def _worker(gpu_id, task_queue) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("relax.worker")

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cpu"

    log.info(f"Loading UMA-M OMAT on {device}...")
    calc = FAIRChemCalculator.from_model_checkpoint(
        "uma-m-1p1", task_name="omat", device=device
    )
    while True:
        item = task_queue.get()
        if item is None:
            break
        name, poscar_path = item
        _relax_one(name, Path(poscar_path), calc)


def _detect_gpus() -> list:
    if not torch.cuda.is_available():
        return []
    eligible = []
    for i in range(torch.cuda.device_count()):
        free_gb = torch.cuda.mem_get_info(i)[0] / 1e9
        total_gb = torch.cuda.mem_get_info(i)[1] / 1e9
        ok = free_gb >= MIN_FREE_VRAM_GB
        status = "OK" if ok else f"LOW VRAM ({free_gb:.1f} GB free) – skipped"
        name = torch.cuda.get_device_properties(i).name
        print(f"  GPU {i}: {name}  {free_gb:.1f}/{total_gb:.1f} GB  [{status}]")
        if ok:
            eligible.append((free_gb, i))
    eligible.sort(reverse=True)
    return [i for _, i in eligible]


def _write_manifest(tasks: list) -> None:
    rows = []
    for name, _ in tasks:
        traj = UMA_RELAXED / f"{name}.traj"
        if traj.exists():
            rows.append({
                "slab_name": name,
                "slab_file": str(traj),
                "millers":   str(_parse_millers(name)),
            })
    pd.DataFrame(rows).to_csv(MANIFEST_CSV, index=False)
    print(f"Manifest written: {MANIFEST_CSV} ({len(rows)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Relax POSCARs with UMA-M OMAT for AdsorbML")
    parser.add_argument("--include", type=str, default=None,
                        help="Comma-separated glob patterns to filter structures")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: one per eligible GPU)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    UMA_RELAXED.mkdir(parents=True, exist_ok=True)

    include_patterns = [p.strip() for p in args.include.split(",")] if args.include else None
    all_structures = discover_structures(DATA_INPUTS, include_patterns=include_patterns)

    tasks = [
        (s[2].name, s[2] / "POSCAR")
        for s in all_structures
        if not _is_excluded(s[2].name)
    ]
    pending = [(name, path) for name, path in tasks
               if not (UMA_RELAXED / f"{name}.traj").exists()]

    print(f"Structures: {len(tasks)} total  |  "
          f"{len(tasks) - len(pending)} done  |  {len(pending)} pending")

    if pending:
        print("Detecting GPUs...")
        gpu_ids = _detect_gpus()
        if not gpu_ids:
            print("No eligible GPU — running on CPU (slow).")
            gpu_ids = [None]
        else:
            print(f"Eligible GPUs: {gpu_ids}")

        n_workers = args.workers or (len(gpu_ids) * WORKERS_PER_GPU)
        ctx   = mp.get_context("spawn")
        queue = ctx.Queue()

        for item in pending:
            queue.put((item[0], str(item[1])))
        for _ in range(n_workers):
            queue.put(None)

        procs = [
            ctx.Process(target=_worker, args=(gpu_ids[i % len(gpu_ids)], queue))
            for i in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    _write_manifest(tasks)


if __name__ == "__main__":
    main()