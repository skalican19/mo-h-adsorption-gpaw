"""
scripts/adsorbml/2-run_adsorbml.py

AdsorbML step 2 (map: screen): Screen H* adsorption candidates on UMA-M relaxed
slabs using AdsorbML (NUM_PLACEMENTS placements per slab, ML-relaxed with the
UMA-M OC20 head).

Relaxations use BestFrameLBFGS (see _common.py), so each candidate keeps its
lowest-fmax frame and reports whether it converged; plain LBFGS keeps the last frame,
which is not necessarily the best one.

Reads:  <data>/adsorbml_manifest.csv
Writes: <data>/adsorbml_results/<name>/candidates.csv   (+ Fmax/converged/nsteps cols)
        <data>/adsorbml_results/<name>/candidate_*.traj (now carry energy + forces)
        <data>/adsorbml_results/<name>/anomalies.csv    (why placements were rejected)
        <data>/adsorbml_results/<name>/adsorbml.log
        <data>/adsorbml_results/batch_summary.csv   (reduce; single run only)

`<data>` is the repo's data/ dir, or $ADSORBML_DATA_ROOT if set (HPC scratch).

Usage (local — one command; batch summary written automatically):
  python scripts/adsorbml/2-run_adsorbml.py
  python scripts/adsorbml/2-run_adsorbml.py --overwrite   # re-screen slabs already done

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

import numpy as np
import pandas as pd
import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

# fairchem (and therefore torch) is imported lazily inside process_row(), not here.
# run_gpu_workers() spawns workers with multiprocessing's 'spawn' context, which
# re-imports this module in every child process before _worker_loop() (_common.py)
# sets CUDA_VISIBLE_DEVICES. A module-level fairchem import would touch/initialize
# CUDA during that re-import, before the per-worker GPU pin is applied, so every
# worker's pin would silently no-op and they'd all default to cuda:0 -- the same
# physical GPU regardless of how many were allocated. See 1-relax_uma_omat.py /
# _common.py's own lazy fairchem import for the same reason.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    MANIFEST_CSV, OUT_DIR, ADSORBATE_SMILES, FMAX, MAX_STEPS_PLACEMENT,
    NUM_PLACEMENTS, BestFrameLBFGS,
    setup_logging, parse_millers, get_shard, apply_shard,
    write_atomic_csv, run_gpu_workers,
)

_CANDIDATES_COLS = [
    "candidate_rank", "E_adslab_ml_eV", "E_slab_ml_eV",
    "E_gas_ref_ml_eV", "E_ads_ml_eV", "anomalies", "traj_path",
    # Relaxation quality of this candidate (from BestFrameLBFGS via atoms.info).
    # 'anomalies' above is kept for schema compatibility but is always empty:
    # run_adsorbml only returns anomaly-FREE candidates. The rejected placements
    # are recorded separately in anomalies.csv.
    "Fmax_adslab_eV_per_Ang", "adslab_converged", "adslab_nsteps", "adslab_best_step",
]

_ANOMALIES_COLS = ["placement_index", "anomalies"]

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


def process_row(item, calc) -> None:
    """Worker adapter: unpack a (manifest_row, overwrite) queue item."""
    from fairchem.data.oc.core import Slab
    from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml

    row, overwrite = item
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

    if done_csv.exists() and not overwrite:
        comp_log.info(f"SKIP (already done): {slab_name}")
        _close_log(comp_log)
        return
    if done_csv.exists():
        # Clear the previous run's outputs. candidates.csv goes FIRST because it is the
        # done-marker _is_done() keys on: if this run is then killed (walltime, crash),
        # the slab reads as not-done and gets redone, rather than being marked complete
        # while its candidate_*.traj files are already gone. The traj sweep is needed
        # because a shorter candidate list this time would otherwise leave orphaned
        # candidate_<k>.traj files that candidates.csv no longer references.
        done_csv.unlink()
        stale = sorted(run_dir.glob("candidate_*.traj")) + [run_dir / "anomalies.csv"]
        for p in stale:
            if p.exists():
                p.unlink()
        comp_log.info(f"OVERWRITE: recomputing {slab_name} "
                      f"(removed candidates.csv + {len(stale)} previous output file(s))")

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
            # BestFrameLBFGS instead of plain LBFGS: fairchem's relax_job discards the
            # convergence bool and keeps the LAST frame, which (no line search) can be
            # worse than the input. BestFrameLBFGS restores the lowest-fmax frame into
            # the live atoms before relax_job reads energy/forces off it, and records
            # the flags in atoms.info.
            optimizer_cls=BestFrameLBFGS,
            fmax=FMAX,
            steps=MAX_STEPS_PLACEMENT,
            num_placements=NUM_PLACEMENTS,
            reference_ml_energies=True,
        )
    except Exception as exc:
        comp_log.error(f"run_adsorbml failed: {exc}\n{traceback.format_exc()}")
        _close_log(comp_log)
        return

    # Record which placements were REJECTED and why. outputs["adslab_anomalies"] covers
    # all NUM_PLACEMENTS placements in generation order, while outputs["adslabs"] holds
    # only the survivors — so this is the only place the rejection reasons exist.
    all_anomalies = outputs.get("adslab_anomalies") or []
    write_atomic_csv(
        pd.DataFrame(
            [{"placement_index": i, "anomalies": "|".join(a)} for i, a in enumerate(all_anomalies)],
            columns=_ANOMALIES_COLS,
        ),
        run_dir / "anomalies.csv",
    )

    candidates = outputs["adslabs"]
    n_rejected = sum(1 for a in all_anomalies if a)
    comp_log.info(
        f"Placements: {len(all_anomalies)} relaxed, {n_rejected} rejected by anomaly "
        f"detection, {len(candidates)} kept"
    )
    if not candidates:
        comp_log.warning(f"No valid placements for {slab_name}")
        write_atomic_csv(pd.DataFrame(columns=_CANDIDATES_COLS), done_csv)
        _close_log(comp_log)
        return

    rows = []
    for i, cand in enumerate(candidates):
        traj_path = str(run_dir / f"candidate_{i}.traj")
        atoms = cand["atoms"]
        res   = cand["results"]

        # fairchem's relax_job sets atoms.calc = None before returning, so the energy
        # and forces must come from res — reading them off atoms raises. (That is why
        # every candidate traj written before 2026-08 has no forces and no fmax.)
        energy, forces = res.get("energy"), res.get("forces")
        if energy is None or forces is None:
            comp_log.warning(f"candidate {i}: missing energy/forces in results; "
                             f"traj written without a calculator")
            fmax = float("nan")
        else:
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            fmax = float(np.sqrt((np.asarray(forces) ** 2).sum(axis=1)).max())

        ase.io.write(traj_path, atoms)
        ref  = res.get("referenced_adsorption_energy", {})
        info = atoms.info
        rows.append({
            "candidate_rank":  i,
            "E_adslab_ml_eV":  res.get("energy", float("nan")),
            "E_slab_ml_eV":    ref.get("slab_energy", float("nan")),
            "E_gas_ref_ml_eV": ref.get("gas_reactant_energy", float("nan")),
            "E_ads_ml_eV":     ref.get("adsorption_energy", float("nan")),
            "anomalies":       "|".join(res.get("adslab_anomalies", [])),
            "traj_path":       traj_path,
            "Fmax_adslab_eV_per_Ang": fmax,
            "adslab_converged":       info.get("relax_converged"),
            "adslab_nsteps":          info.get("relax_nsteps"),
            "adslab_best_step":       info.get("relax_best_step"),
        })

    write_atomic_csv(pd.DataFrame(rows, columns=_CANDIDATES_COLS), done_csv)

    best_e = min(r["E_ads_ml_eV"] for r in rows)
    n_conv = sum(1 for r in rows if r["adslab_converged"] is True)
    fmaxes = [r["Fmax_adslab_eV_per_Ang"] for r in rows
              if not pd.isna(r["Fmax_adslab_eV_per_Ang"])]
    median_fmax = f"{np.median(fmaxes):.4f} eV/Å" if fmaxes else "n/a"
    comp_log.info(f"Best E_ads (ML) = {best_e:.4f} eV  ({len(candidates)} candidates)")
    comp_log.info(f"Converged candidates (fmax <= {FMAX} eV/Å): {n_conv}/{len(rows)}  |  "
                  f"median fmax {median_fmax}")
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
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-screen even if <slab>/candidates.csv already exists, replacing "
                             "that slab's candidate_*.traj and anomalies.csv. Without this, "
                             "screened slabs are skipped, so a methodology change has no effect "
                             "on slabs already done.")
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
    pending  = [r for r in all_rows if args.overwrite or not _is_done(r["slab_name"])]
    shard    = get_shard(args.shard)
    pending  = apply_shard(pending, shard)
    master_log.info(
        f"Total: {len(all_rows)}  |  shard {shard[0]}/{shard[1]}  |  "
        f"Pending in this shard: {len(pending)}"
        f"{'  |  OVERWRITE: existing results will be replaced' if args.overwrite else ''}"
    )

    run_gpu_workers([(r, args.overwrite) for r in pending],
                    task_name="oc20", process_item=process_row, n_workers=args.workers)

    if shard[1] == 1:
        _write_batch_summary()
    else:
        master_log.info("Multi-shard run: batch_summary NOT written. Run "
                        "`python scripts/adsorbml/2-run_adsorbml.py --summary-only` afterward (optional).")


if __name__ == "__main__":
    main()
