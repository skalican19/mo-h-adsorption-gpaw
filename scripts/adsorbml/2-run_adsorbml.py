"""
scripts/adsorbml/2-run_adsorbml.py

AdsorbML step 2: Screen H* adsorption candidates on UMA-M relaxed slabs using
AdsorbML (100 placements per slab, ML-relaxed with UMA-M OC20 head).

Reads:  data/adsorbml_manifest.csv
Writes: data/adsorbml_results/<name>/candidates.csv
        data/adsorbml_results/<name>/candidate_*.traj
        data/adsorbml_results/<name>/adsorbml.log
        data/adsorbml_results/batch_summary.csv

Usage:
  python scripts/adsorbml/2-run_adsorbml.py
"""
import ast
import glob
import logging
import os
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import ase.io
from ase.optimize import LBFGS
from fairchem.data.oc.core import Slab
from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml
from fairchem.core import FAIRChemCalculator

REPO_ROOT        = Path(__file__).resolve().parents[2]
MANIFEST_CSV     = REPO_ROOT / "data" / "adsorbml_manifest.csv"
OUT_DIR          = REPO_ROOT / "data" / "adsorbml_results"
ADSORBATE_SMILES = "*H"
MIN_FREE_VRAM_GB = 8.0
WORKERS_PER_GPU  = 1

_CANDIDATES_COLS = [
    "candidate_rank", "E_adslab_ml_eV", "E_slab_ml_eV",
    "E_gas_ref_ml_eV", "E_ads_ml_eV", "anomalies", "traj_path",
]

master_log = logging.getLogger("adsorbml.master")
_LOG_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_COMPOUND_LOG_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def _setup_logging(log_path=None) -> None:
    master_log.setLevel(logging.DEBUG)
    if master_log.handlers:
        return
    ch = logging.StreamHandler()
    ch.setFormatter(_LOG_FMT)
    master_log.addHandler(ch)
    if log_path:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(_LOG_FMT)
        master_log.addHandler(fh)


def _parse_millers(millers_str) -> tuple:
    if isinstance(millers_str, tuple):
        return millers_str
    try:
        v = ast.literal_eval(str(millers_str))
        return v if isinstance(v, tuple) else (0, 0, 1)
    except Exception:
        return (0, 0, 1)


def _is_done(slab_name: str) -> bool:
    return (OUT_DIR / slab_name / "candidates.csv").exists()


def _close_log(log: logging.Logger) -> None:
    for h in log.handlers[:]:
        h.close()
        log.removeHandler(h)


def process_row(row: dict, calc) -> None:
    slab_file = row["slab_file"]
    slab_name = row["slab_name"]
    millers   = _parse_millers(row["millers"])

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
            fmax=0.02,
            steps=200,
            num_placements=100,
            reference_ml_energies=True,
            place_on_relaxed_slab=False,
        )
    except Exception as exc:
        comp_log.error(f"run_adsorbml failed: {exc}")
        _close_log(comp_log)
        return

    candidates = outputs["adslabs"]
    if not candidates:
        comp_log.warning(f"No valid placements for {slab_name}")
        pd.DataFrame(columns=_CANDIDATES_COLS).to_csv(done_csv, index=False)
        _close_log(comp_log)
        return

    rows = []
    for i, cand in enumerate(candidates):
        traj_path = str(run_dir / f"candidate_{i}.traj")
        ase.io.write(traj_path, cand["atoms"])
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

    pd.DataFrame(rows).to_csv(done_csv, index=False)
    best_e = min(r["E_ads_ml_eV"] for r in rows)
    comp_log.info(f"Best E_ads (ML) = {best_e:.4f} eV  ({len(candidates)} candidates)")
    _close_log(comp_log)


def _detect_gpus() -> list:
    if not torch.cuda.is_available():
        return []
    eligible = []
    master_log.info("GPU inventory:")
    for i in range(torch.cuda.device_count()):
        props    = torch.cuda.get_device_properties(i)
        free_gb  = torch.cuda.mem_get_info(i)[0] / 1e9
        total_gb = torch.cuda.mem_get_info(i)[1] / 1e9
        ok = free_gb >= MIN_FREE_VRAM_GB
        status = "OK" if ok else f"LOW VRAM – skipped"
        master_log.info(f"  GPU {i}: {props.name}  {free_gb:.1f}/{total_gb:.1f} GB  [{status}]")
        if ok:
            eligible.append((free_gb, i))
    eligible.sort(reverse=True)
    return [i for _, i in eligible]


def _worker(gpu_id, worker_idx: int, task_queue) -> None:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cpu"
    master_log.info(f"Loading UMA-M OC20 on {device} (worker {worker_idx})...")
    calc = FAIRChemCalculator.from_model_checkpoint("uma-m-1p1", task_name="oc20", device=device)
    while True:
        row = task_queue.get()
        if row is None:
            break
        process_row(row, calc)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(OUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    _setup_logging(log_path)
    master_log.info(f"Log: {log_path}")

    master_log.info("Detecting GPUs...")
    gpu_ids  = _detect_gpus()
    n_workers = len(gpu_ids) * WORKERS_PER_GPU if gpu_ids else 1
    if gpu_ids:
        master_log.info(f"Launching {n_workers} worker(s) across GPU(s): {gpu_ids}")
    else:
        master_log.warning("No eligible GPU — running on CPU.")

    df       = pd.read_csv(MANIFEST_CSV)
    all_rows = df.to_dict("records")
    pending  = [r for r in all_rows if not _is_done(r["slab_name"])]
    master_log.info(
        f"Total: {len(all_rows)}  |  Done: {len(all_rows) - len(pending)}  |  Pending: {len(pending)}"
    )

    if not gpu_ids:
        gpu_ids = [None]

    if pending:
        ctx   = mp.get_context("spawn")
        queue = ctx.Queue()
        for row in pending:
            queue.put(row)
        for _ in range(n_workers):
            queue.put(None)

        procs = [
            ctx.Process(target=_worker, args=(gpu_ids[i % len(gpu_ids)], i, queue))
            for i in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    # Consolidate all per-slab candidates.csv into one batch summary
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
        summary      = pd.concat(frames, ignore_index=True)
        summary_path = OUT_DIR / "batch_summary.csv"
        summary.to_csv(summary_path, index=False)
        master_log.info(f"Saved consolidated results → {summary_path}")

        best = summary.loc[summary.groupby("slab_name")["E_ads_ml_eV"].idxmin(),
                           ["slab_name", "E_ads_ml_eV"]]
        master_log.info("Best ML adsorption energies per slab:\n" + best.to_string(index=False))
    else:
        master_log.info("No results to summarise yet.")
