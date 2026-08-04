"""
scripts/adsorbml/1-relax_uma_omat.py

AdsorbML step 1 (map: relax): Relax POSCAR structures with UMA-M (OMAT head) and
build a manifest CSV for the next step.

Excluded: structures with 'graphene', 'nanoribbon', or 'edge' in their name.

Outputs:
  <data>/uma_relaxed/<name>.traj   — relaxed slab per structure
  <data>/adsorbml_manifest.csv     — manifest for 2-run_adsorbml.py

`<data>` is the repo's data/ dir, or $ADSORBML_DATA_ROOT if set (HPC scratch).

Usage (local — one command; manifest written automatically):
  python scripts/adsorbml/1-relax_uma_omat.py
  python scripts/adsorbml/1-relax_uma_omat.py --include "MoS2_*,Mo2N_*"
  python scripts/adsorbml/1-relax_uma_omat.py --workers 2

Usage (HPC SLURM array — shard the relax, then build the manifest ONCE):
  # each array task, e.g. sbatch --array=0-7 :
  python scripts/adsorbml/1-relax_uma_omat.py --shard $SLURM_ARRAY_TASK_ID/8
  # after the array finishes, a single reduce job:
  python scripts/adsorbml/1-relax_uma_omat.py --manifest-only
"""
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from ase.optimize import LBFGS
from ase.constraints import FixAtoms

# scripts/ on the path makes "adsorbml" resolve as a namespace package, so
# adsorbml._common and _common get distinct sys.modules keys (no basename clash).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import discover_structures
from adsorbml._common import (
    DATA_INPUTS, UMA_RELAXED, MANIFEST_CSV, FMAX, MAX_STEPS,
    setup_logging, millers_from_name, get_shard, apply_shard,
    write_atomic_csv, write_atomic_traj, run_gpu_workers,
)

# Structures excluded from AdsorbML (not 2D-periodic surface slabs or off-topic)
_EXCLUDE = ("graphene", "nanoribbon", "edge")


def _is_excluded(name: str) -> bool:
    low = name.lower()
    return any(pat in low for pat in _EXCLUDE)


def _tile_slab(atoms):
    """Repeat the slab in a and b until both cell dimensions are >= 8 Å (required by ocp_adslab_generator)."""
    cell = atoms.get_cell()
    na = max(1, int(np.ceil(8.0 / np.linalg.norm(cell[0]))))
    nb = max(1, int(np.ceil(8.0 / np.linalg.norm(cell[1]))))
    return atoms.repeat([na, nb, 1]) if (na > 1 or nb > 1) else atoms


def _tag_atoms(atoms):
    """Assign OC20 surface tags: 1=surface layer, 0=subsurface/bulk. tag=2 is reserved for adsorbates."""
    z_max = atoms.positions[:, 2].max()
    tags = [1 if atom.position[2] > z_max - 2.0 else 0 for atom in atoms]
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
        atoms = _tile_slab(atoms)
        atoms = _tag_atoms(atoms)
        atoms.set_constraint(FixAtoms(mask=[t == 0 for t in atoms.get_tags()]))
        atoms.calc = calc
        opt = LBFGS(atoms, logfile=str(UMA_RELAXED / f"{name}_opt.log"))
        opt.run(fmax=FMAX, steps=MAX_STEPS)
        write_atomic_traj(atoms, out_traj)
        log.info(f"Done: {name}  E={atoms.get_potential_energy():.4f} eV")
    except Exception as exc:
        log.error(f"Failed {name}: {exc}")


def _relax_task(item, calc) -> None:
    """Worker adapter: unpack a (name, poscar_path) queue item."""
    name, poscar_path = item
    _relax_one(name, Path(poscar_path), calc)


def _write_manifest(tasks: list) -> None:
    rows = []
    for name, _ in tasks:
        traj = UMA_RELAXED / f"{name}.traj"
        if traj.exists():
            rows.append({
                "slab_name": name,
                "slab_file": str(traj),
                "millers":   str(millers_from_name(name)),
            })
    write_atomic_csv(pd.DataFrame(rows), MANIFEST_CSV)
    print(f"Manifest written: {MANIFEST_CSV} ({len(rows)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Relax POSCARs with UMA-M OMAT for AdsorbML")
    parser.add_argument("--include", type=str, default=None,
                        help="Comma-separated glob patterns to filter structures")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: one per eligible GPU)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process a disjoint stride I/N of the pending list "
                             "(default: from SLURM_ARRAY_TASK_ID, else the whole set).")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Skip relaxation; just (re)build the manifest from existing trajs.")
    args = parser.parse_args()

    setup_logging()
    UMA_RELAXED.mkdir(parents=True, exist_ok=True)

    include_patterns = [p.strip() for p in args.include.split(",")] if args.include else None
    all_structures = discover_structures(DATA_INPUTS, include_patterns=include_patterns)
    tasks = sorted(
        ((s[2].name, s[2] / "POSCAR") for s in all_structures if not _is_excluded(s[2].name)),
        key=lambda t: t[0],
    )

    if args.manifest_only:
        _write_manifest(tasks)
        return

    pending = [(name, path) for name, path in tasks
               if not (UMA_RELAXED / f"{name}.traj").exists()]
    shard = get_shard(args.shard)
    pending = apply_shard(pending, shard)

    print(f"Structures: {len(tasks)} total  |  shard {shard[0]}/{shard[1]}  |  "
          f"{len(pending)} to run in this shard")

    run_gpu_workers(
        [(name, str(path)) for name, path in pending],
        task_name="omat",
        process_item=_relax_task,
        n_workers=args.workers,
    )

    if shard[1] == 1:
        _write_manifest(tasks)
    else:
        print("Multi-shard run: manifest NOT written. Run "
              "`python scripts/adsorbml/1-relax_uma_omat.py --manifest-only` "
              "once the array completes.")


if __name__ == "__main__":
    main()
