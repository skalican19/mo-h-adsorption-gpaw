"""
scripts/adsorbml/2-run_adsorbml.py

AdsorbML step 2 (map: screen): Screen H* adsorption candidates on UMA-M relaxed
slabs using AdsorbML (NUM_PLACEMENTS placements per slab, ML-relaxed with the
UMA-M OC20 head).

Reads:  <data>/adsorbml_manifest.csv
Writes: <data>/adsorbml_results/<name>/candidates.csv
        <data>/adsorbml_results/<name>/candidate_*.traj
        <data>/adsorbml_results/<name>/adsorbml.log
        <data>/adsorbml_results/batch_summary.csv   (reduce; single run only)

`<data>` is the repo's data/ dir, or $ADSORBML_DATA_ROOT if set (HPC scratch).

Usage (local — one command; batch summary written automatically):
  python scripts/adsorbml/2-run_adsorbml.py

Usage (HPC SLURM array — shard the screening, then summarise ONCE):
  python scripts/adsorbml/2-run_adsorbml.py --shard $SLURM_ARRAY_TASK_ID/8
  python scripts/adsorbml/2-run_adsorbml.py --summary-only    # optional reduce
"""
import argparse
import glob
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
from fairchem.data.oc.core import Slab
from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    MANIFEST_CSV, OUT_DIR, ADSORBATE_SMILES, FMAX, MAX_STEPS, NUM_PLACEMENTS,
    setup_logging, parse_millers, get_shard, apply_shard,
    write_atomic_csv, run_gpu_workers,
)

_CANDIDATES_COLS = [
    "candidate_rank", "E_adslab_ml_eV", "E_slab_ml_eV",
    "E_gas_ref_ml_eV", "E_ads_ml_eV", "anomalies", "traj_path",
]

master_log = logging.getLogger("adsorbml.master")
# Per-slab file/console logging uses its own formatters (kept local to step 2).
_LOG_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_COMPOUND_LOG_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def _is_done(slab_name: str) -> bool:
    return (OUT_DIR / slab_name / "candidates.csv").exists()


def _close_log(log: logging.Logger) -> None:
    for h in log.handlers[:]:
        h.close()
        log.removeHandler(h)


def process_row(row: dict, calc) -> None:
    slab_file = row["slab_file"]
    slab_name = row["slab_name"]
    millers   = parse_millers(row["millers"])

    run_dir  = OUT_DIR / slab_name
    done_csv = run_dir / "candidates.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    comp_log = logging.getLogger(f"adsorbml.{slab_name}")
    comp_log.setLevel(logging.DEBUG)
    comp_log.propagate = False
    if not comp_log.handlers:
        fh = logging.FileHandler(str(run_dir / "adsorbml.log"), mode="a", encoding="utf-8")
        fh.setFormatter(_LOG_FMT)
        comp_log.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(_COMPOUND_LOG_FMT)
        comp_log.addHandler(ch)

    if done_csv.exists():
        comp_log.info(f"SKIP (already done): {slab_name}")
        _close_log(comp_log)
        return

    comp_log.info("=" * 50)
    comp_log.info(f"Running: {slab_name}")

    try:
        atoms = ase.io.read(slab_file)
        if not atoms.constraints:
            atoms.set_constraint(FixAtoms(mask=[t == 0 for t in atoms.get_tags()]))
        slab  = Slab(bulk=None, slab_atoms=atoms, millers=millers,
                     shift=None, top=None, oriented_bulk=None)
    except Exception as exc:
        comp_log.error(f"Could not load slab: {exc}")
        _close_log(comp_log)
        return

    try:
        outputs = run_adsorbml(
            slab=slab,
            adsorbate=ADSORBATE_SMILES,
            calculator=calc,
            optimizer_cls=LBFGS,
            fmax=FMAX,
            steps=MAX_STEPS,
            num_placements=NUM_PLACEMENTS,
            reference_ml_energies=True,
        )
    except Exception as exc:
        comp_log.error(f"run_adsorbml failed: {exc}\n{traceback.format_exc()}")
        _close_log(comp_log)
        return

    candidates = outputs["adslabs"]
    if not candidates:
        comp_log.warning(f"No valid placements for {slab_name}")
        write_atomic_csv(pd.DataFrame(columns=_CANDIDATES_COLS), done_csv)
        _close_log(comp_log)
        return

    rows = []
    for i, cand in enumerate(candidates):
        traj_path = str(run_dir / f"candidate_{i}.traj")
        atoms = cand["atoms"]
        try:
            spc = SinglePointCalculator(
                atoms,
                energy=atoms.get_potential_energy(),
                forces=atoms.get_forces(),
            )
            atoms.calc = spc
        except Exception:
            pass
        ase.io.write(traj_path, atoms)
        res = cand["results"]
        ref = res.get("referenced_adsorption_energy", {})
        rows.append({
            "candidate_rank":  i,
            "E_adslab_ml_eV":  res.get("energy", float("nan")),
            "E_slab_ml_eV":    ref.get("slab_energy", float("nan")),
            "E_gas_ref_ml_eV": ref.get("gas_reactant_energy", float("nan")),
            "E_ads_ml_eV":     ref.get("adsorption_energy", float("nan")),
            "anomalies":       "|".join(res.get("adslab_anomalies", [])),
            "traj_path":       traj_path,
        })

    write_atomic_csv(pd.DataFrame(rows), done_csv)
    best_e = min(r["E_ads_ml_eV"] for r in rows)
    comp_log.info(f"Best E_ads (ML) = {best_e:.4f} eV  ({len(candidates)} candidates)")
    _close_log(comp_log)


def _write_batch_summary() -> None:
    """Reduce: consolidate all per-slab candidates.csv into one batch summary."""
    all_csvs = sorted(glob.glob(str(OUT_DIR / "*" / "candidates.csv")))
    frames = []
    for csv_path in all_csvs:
        try:
            part = pd.read_csv(csv_path)
            if len(part) > 0:
                part.insert(0, "slab_name", Path(csv_path).parent.name)
                frames.append(part)
        except Exception as exc:
            master_log.warning(f"Skipping {csv_path}: {exc}")

    if frames:
        summary = pd.concat(frames, ignore_index=True)
        summary_path = OUT_DIR / "batch_summary.csv"
        write_atomic_csv(summary, summary_path)
        master_log.info(f"Saved consolidated results → {summary_path}")
        best = summary.loc[summary.groupby("slab_name")["E_ads_ml_eV"].idxmin(),
                           ["slab_name", "E_ads_ml_eV"]]
        master_log.info("Best ML adsorption energies per slab:\n" + best.to_string(index=False))
    else:
        master_log.info("No results to summarise yet.")


def main():
    parser = argparse.ArgumentParser(description="Screen H* sites on relaxed slabs (AdsorbML)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: one per eligible GPU)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process a disjoint stride I/N of the pending slabs "
                             "(default: from SLURM_ARRAY_TASK_ID, else the whole set).")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip screening; just (re)build batch_summary.csv from existing candidates.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(OUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_path)
    master_log.info(f"Log: {log_path}")

    if args.summary_only:
        _write_batch_summary()
        return

    df       = pd.read_csv(MANIFEST_CSV)
    all_rows = sorted(df.to_dict("records"), key=lambda r: r["slab_name"])
    pending  = [r for r in all_rows if not _is_done(r["slab_name"])]
    shard    = get_shard(args.shard)
    pending  = apply_shard(pending, shard)
    master_log.info(
        f"Total: {len(all_rows)}  |  shard {shard[0]}/{shard[1]}  |  "
        f"Pending in this shard: {len(pending)}"
    )

    run_gpu_workers(pending, task_name="oc20", process_item=process_row, n_workers=args.workers)

    if shard[1] == 1:
        _write_batch_summary()
    else:
        master_log.info("Multi-shard run: batch_summary NOT written. Run "
                        "`python scripts/adsorbml/2-run_adsorbml.py --summary-only` afterward (optional).")


if __name__ == "__main__":
    main()
