# CLAUDE.md — mo-h-adsorption-gpaw

Repo summary for coding agents. Complements `README.md` (human quick-start) with
the internal structure, pipelines, and conventions. Verified 2026-07-01; re-check
against code before relying on specifics.

## What this is

Computes hydrogen adsorption free energy **ΔG_H** on Mo-based catalyst surfaces
(MoS₂, MoSe₂, MoP, Mo₂N, Mo₂C, …; slabs + edges, vacancies, dopants, Ni/Mo
interfaces) to screen HER (hydrogen-evolution) candidates via the Sabatier
criterion (|ΔG_H| → 0 is best).

Core relation: `ΔG_H = E(slab+H) − E(slab) − ½·E(H₂) + 0.24 eV` (0.24 = ZPE+entropy).

There are **two compute paths**, both through `scripts/gpaw_h_adsorption.py` plus
the `scripts/adsorbml/` package:

1. **Standard GPAW DFT** — generate POSCARs → GPAW relax/single-point → ΔG_H.
2. **AdsorbML ML screening** — fairchem UMA-M relaxes slabs and screens H* sites
   fast, then GPAW re-evaluates the top candidates (`--adsorbml-candidates`).

(A `--validate-uma` UMA-vs-DFT benchmark mode existed but was removed; recover
from git history if the Paper-B validation work resumes.)

## Layout

```
scripts/
  generate_structures.py     # build all POSCARs under data/inputs/VASP_inputs/<name>/
  gpaw_h_adsorption.py        # PRIMARY calculator; 3 modes (see CLI flags below)
  adsorbml/                   # ML screening pipeline (steps 1-3) + _common.py
  compute_h_adsorption.py     # LEGACY: Materials Project + VASP input templating
  parse_vasp_results.py       # LEGACY: parse VASP OUTCARs -> ΔG_H
  *.sh                        # HPC (DEVANA/SLURM) + desktop launchers & workers
estimate_progress.py          # live progress/ETA monitor for long GPAW campaigns
data/
  inputs/VASP_inputs/<name>/POSCAR   # inputs (version-controlled, stay in repo)
  outputs/                    # gpaw_h_adsorption_results*.csv/.json, gpaw_calculations/ logs,
                              #   h2_reference_energy.json (cached H₂ ref, config-keyed)
  uma_relaxed/<name>.traj     # AdsorbML step-1 relaxed slabs
  adsorbml_manifest.csv       # AdsorbML step-1 -> step-2 handoff
  adsorbml_results/<name>/    # candidates.csv + candidate_*.traj; ranked_candidates*.csv
requirements.txt              # ase, numpy, pandas, gpaw, pymatgen, fairchem-core, fairchem-data-oc
```

## Key settings (in `scripts/gpaw_h_adsorption.py`, top of file)

- `GPAW_CONFIG`: LCAO / DZP basis / **PBE** / kpts (4,4,1).
- `RELAXATION_CONFIG`: fmax 0.10 eV/Å, 8 steps (coarse; overridable via `--fmax`/`--relax-steps`).
- `ENTROPY_CORRECTION = 0.24 eV`; `CORES_PER_CALC = 11`; `RAM_PER_CALC_GB = 4`.
- AdsorbML step constants live in `scripts/adsorbml/_common.py` (FMAX 0.02, MAX_STEPS 100,
  NUM_PLACEMENTS 100, UMA_MODEL "uma-m-1p1", same 0.24 correction).

## Running

Interpreter: use the pyenv env with the deps — **`~/.pyenv/versions/cemea-env/bin/python`**
(has ase/torch/gpaw/fairchem). Bare `python` is not on PATH.

**Standard GPAW** (auto-discovers all POSCARs; incremental, crash-safe CSV):
```
python scripts/gpaw_h_adsorption.py [--include "Mo2N_*"] [--structure-name NAME]
       [--workers N] [--no-relax] [--kpts 4,4,1] [--machine node1|devana]
# HPC array (one structure/task): ACCOUNT=proj bash scripts/submit_devana_gpaw_array.sh
# Desktop:                        MACHINE=node1 bash scripts/run_desktop_machine.sh
```

**AdsorbML pipeline** (`scripts/adsorbml/`, refactored — see `_common.py`):
```
# 1) relax with UMA-M OMAT      2) screen H* sites (100/slab)   3) rank by |ΔG*H|
python scripts/adsorbml/1-relax_uma_omat.py [--include ...] [--workers N]
python scripts/adsorbml/2-run_adsorbml.py
python scripts/adsorbml/3-extract_rank.py        # -> ranked_candidates_stable.csv
# then GPAW re-check the ranked candidates:
python scripts/gpaw_h_adsorption.py --adsorbml-candidates data/adsorbml_results/ranked_candidates_stable.csv
```

The AdsorbML steps run **locally (single command each) and as concurrent SLURM
arrays**:
- `--shard I/N` (or auto from `SLURM_ARRAY_TASK_ID/COUNT`, contiguous `--array=a-b`)
  splits per-structure work into disjoint strides — no locking, no races.
- Writes are atomic (temp + `os.replace`); safe under walltime kills.
- Steps 1 and 2 are **maps**; their aggregations are **reduces** run once:
  after an array, run `1-...py --manifest-only` and (optional) `2-...py --summary-only`.
  Step 3 is always a single reduce. Locally (shard 0/1) the reduce runs automatically.
- `$ADSORBML_DATA_ROOT` redirects **outputs** to fast scratch (inputs stay in repo);
  steps 1 and 2 must share the same value (manifest stores absolute traj paths).

## Conventions / gotchas

- **Never clobber `data/`.** It holds real, expensive results (hundreds of trajs,
  candidates, ranked CSVs). Test destructive paths against a temp `$ADSORBML_DATA_ROOT`
  or a copied subset.
- Results CSVs are append/checkpoint style — reruns resume by skipping completed rows
  (keyed by structure, or output-file existence in AdsorbML).
- H₂ reference energy is cached in `data/outputs/h2_reference_energy.json` and invalidated
  on XC/config change; UMA mode uses a separate `h2_reference_validation.json`.
- Legacy VASP scripts (`compute_h_adsorption.py`, `parse_vasp_results.py`) are not the
  primary path; GPAW is.
- Known data caveats and history are tracked in the agent memory dir (see MEMORY.md):
  the LDA→PBE switch, Mo₂N (111) vacancy generation bug, and that the old gpaw ΔG_H CSV
  is not trustworthy ground truth.

## Status / direction (from README, 2026-02-11)

MoS₂/MoSe₂ basal planes are inert → focus on **edges and defects**. **Mo₂N** is the
strongest candidate; refine with dopants/vacancies. Use AdsorbML/UMA to shortlist,
GPAW to compute, VASP only for final validation of top candidates.
