"""
scripts/adsorbml/1-relax_uma_omat.py

AdsorbML step 1 (map: relax): Relax POSCAR structures with UMA-M (OMAT head) and
build a manifest CSV for the next step.

Excluded: structures with 'graphene', 'nanoribbon', or 'edge' in their name.

Relaxation uses BestFrameLBFGS (see adsorbml/_common.py): the lowest-fmax frame is
kept rather than the last one, and convergence flags land in atoms.info and in the
manifest, so "converged" and "ran out of steps" stay distinguishable downstream.

Outputs:
  <data>/uma_relaxed/<name>.traj   — relaxed slab per structure (+ relax_* in atoms.info)
  <data>/uma_relaxed/<name>_opt.log — per-step optimizer log
  <data>/adsorbml_manifest.csv     — manifest for 2-run_adsorbml.py, incl.
                                     relax_converged / relax_nsteps / relax_fmax

`<data>` is the repo's data/ dir, or $ADSORBML_DATA_ROOT if set (HPC scratch).

Usage (local — one command; manifest written automatically):
  python scripts/adsorbml/1-relax_uma_omat.py
  python scripts/adsorbml/1-relax_uma_omat.py --include "MoS2_*,Mo2N_*"
  python scripts/adsorbml/1-relax_uma_omat.py --workers 2
  python scripts/adsorbml/1-relax_uma_omat.py --overwrite   # re-relax slabs already done

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
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator

# scripts/ on the path makes "adsorbml" resolve as a namespace package, so
# adsorbml._common and _common get distinct sys.modules keys (no basename clash).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import discover_structures
from adsorbml._common import (
    DATA_INPUTS, UMA_RELAXED, MANIFEST_CSV, FMAX, MAX_STEPS_SLAB,
    BestFrameLBFGS, relax_outcome,
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


def _relax_one(name: str, poscar_path: Path, calc, overwrite: bool = False) -> None:
    log = logging.getLogger(f"relax.{name}")
    out_traj = UMA_RELAXED / f"{name}.traj"

    if out_traj.exists() and not overwrite:
        log.info(f"SKIP (already done): {name}")
        return
    if out_traj.exists():
        log.info(f"OVERWRITE: recomputing {name} (existing traj will be replaced)")

    log.info(f"Start: {name}")
    try:
        atoms = read(str(poscar_path))
        atoms = _tile_slab(atoms)
        atoms = _tag_atoms(atoms)
        atoms.set_constraint(FixAtoms(mask=[t == 0 for t in atoms.get_tags()]))
        atoms.calc = calc

        # BestFrameLBFGS keeps the lowest-fmax frame (plain LBFGS can end worse than
        # it started) and stamps the convergence flags into atoms.info. On return the
        # live atoms IS the kept frame, so the energy/forces below describe it.
        opt = BestFrameLBFGS(atoms, logfile=str(UMA_RELAXED / f"{name}_opt.log"))
        opt.run(fmax=FMAX, steps=MAX_STEPS_SLAB)

        energy = float(atoms.get_potential_energy())
        snapshot = atoms.copy()                     # keeps tags, constraints, info
        snapshot.calc = SinglePointCalculator(
            snapshot, energy=energy, forces=atoms.get_forces()
        )
        write_atomic_traj(snapshot, out_traj)

        info = atoms.info
        log.info(
            f"Done: {name}  {relax_outcome(info)}  E={energy:.4f} eV  "
            f"fmax={info['relax_fmax']:.4f} eV/Å (target {FMAX}, "
            f"kept step {info['relax_best_step']}/{info['relax_nsteps']}, "
            f"final step fmax={info['relax_fmax_final']:.4f})"
        )
    except Exception as exc:
        log.error(f"Failed {name}: {exc}")


def _relax_task(item, calc) -> None:
    """Worker adapter: unpack a (name, poscar_path, overwrite) queue item."""
    name, poscar_path, overwrite = item
    _relax_one(name, Path(poscar_path), calc, overwrite=overwrite)


def _write_manifest(tasks: list) -> None:
    """Rebuild the manifest from the trajs on disk, carrying the relaxation quality
    flags stamped into atoms.info by _relax_one. Trajs written before those flags
    existed report NaN/None — treat them as unknown, not as converged."""
    rows = []
    for name, _ in tasks:
        traj = UMA_RELAXED / f"{name}.traj"
        if not traj.exists():
            continue
        try:
            info = read(str(traj)).info
        except Exception:
            info = {}
        rows.append({
            "slab_name":       name,
            "slab_file":       str(traj),
            "millers":         str(millers_from_name(name)),
            "relax_converged": info.get("relax_converged"),
            "relax_nsteps":    info.get("relax_nsteps"),
            "relax_fmax":      info.get("relax_fmax", float("nan")),
            "relax_best_step": info.get("relax_best_step"),
        })
    df = pd.DataFrame(rows)
    write_atomic_csv(df, MANIFEST_CSV)
    print(f"Manifest written: {MANIFEST_CSV} ({len(rows)} entries)")
    if len(df):
        conv    = (df["relax_converged"] == True).sum()   # noqa: E712 — None/NaN must not count
        unknown = df["relax_converged"].isna().sum()
        print(f"  converged: {conv}  |  not converged: {len(df) - conv - unknown}  "
              f"|  unknown (pre-flag traj): {unknown}")


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
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if <name>.traj already exists (replaces it). "
                             "Without this, existing trajs are skipped, so a methodology "
                             "change has no effect on structures already relaxed.")
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
               if args.overwrite or not (UMA_RELAXED / f"{name}.traj").exists()]
    shard = get_shard(args.shard)
    pending = apply_shard(pending, shard)

    print(f"Structures: {len(tasks)} total  |  shard {shard[0]}/{shard[1]}  |  "
          f"{len(pending)} to run in this shard"
          f"{'  |  OVERWRITE: existing trajs will be replaced' if args.overwrite else ''}")

    run_gpu_workers(
        [(name, str(path), args.overwrite) for name, path in pending],
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
