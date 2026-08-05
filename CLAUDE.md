# CLAUDE.md — mo-h-adsorption-gpaw

Repo summary for coding agents. Complements `README.md` (human quick-start) with
the internal structure, pipelines, and conventions. Verified 2026-07-09; re-check
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
  _common.py                  # dependency-free helpers shared by BOTH pipelines (discover_structures)
  generate_structures.py     # build all POSCARs under data/inputs/VASP_inputs/<name>/
  gpaw_h_adsorption.py        # PRIMARY calculator; 3 modes (see CLI flags below)
  adsorbml/                   # ML screening pipeline (steps 1-3) + its own _common.py (adsorbml-specific helpers)
  hpc_scripts/adsorbml/       # Perun GPU launchers for the AdsorbML pipeline
                              #   (setup_perun_uma_env.sh, submit_perun_adsorbml.sh, perun_adsorbml_worker.sh)
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
requirements.txt              # includes requirements-gpaw.txt + requirements-adsorbml.txt (union, for the unified dev env)
requirements-gpaw.txt         # ase, numpy, pandas, gpaw, pymatgen — standard DFT path + generate_structures.py
requirements-adsorbml.txt     # ase, numpy, pandas, fairchem-core, fairchem-data-oc — ML screening path (no gpaw)
```

## Key settings (in `scripts/gpaw_h_adsorption.py`, top of file)

- `GPAW_CONFIG`: **PW(350) plane waves** / **RPBE** / `kpts='auto'` (OC20 Monkhorst-Pack
  density `round(40/|a|)` in-plane, 1 in z) / non-spin-polarized / Methfessel-Paxton
  smearing 0.2 eV. Matches the OC20 (RPBE) VASP reference UMA's `oc20` head was trained on.
  (Not comparable to the older LDA/PBE-LCAO ΔG_H CSVs.)
- `RELAXATION_CONFIG`: fmax 0.03 eV/Å, 200 steps (overridable via `--fmax`/`--relax-steps`;
  `--kpts` overrides the mesh). CLI overrides default to None → inherit these config values.
- `ENTROPY_CORRECTION = 0.24 eV`; `CORES_PER_CALC = 11`; `RAM_PER_CALC_GB = 4`.
- AdsorbML step constants live in `scripts/adsorbml/_common.py` (FMAX 0.02,
  MAX_STEPS_SLAB 300 for step 1 / MAX_STEPS_PLACEMENT 100 for step 2, NUM_PLACEMENTS 100,
  UMA_MODEL "uma-m-1p1", same 0.24 correction).
- Both AdsorbML steps relax with **`BestFrameLBFGS`** (`_common.py`), not plain LBFGS:
  ASE's LBFGS has no line search, so the last frame can be worse than the input (in the
  pre-2026-08 data, 50/318 slabs ended worse than they started, one at 214 eV/Å). It keeps
  the lowest-fmax frame and stamps `relax_converged`/`relax_nsteps`/`relax_fmax`/… into
  `atoms.info`, which flow into the manifest, `candidates.csv`, and the ranked CSV.
  **Step 3 reports convergence but does not filter on it** — the selection rule is still
  lowest `E_ads` in the sanity window, so check `Fmax_adslab_eV_per_Ang` /
  `adslab_converged` before trusting the top of a ranking. Results written before
  2026-08 have no flags and must read as *unknown*, never as converged.
- Step 2 also writes `<slab>/anomalies.csv` — why each of the 100 placements was rejected.
  The `anomalies` column in `candidates.csv` is always empty by construction
  (`run_adsorbml` returns only anomaly-free candidates).

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

**AdsorbML on Perun GPU** (aarch64/GH200 nodes; native fairchem venv; steps 1 & 2 only —
rank locally with step 3):
```
# once, on a GPU node: build the aarch64 CUDA fairchem venv, warm the gated uma-m-1p1 cache
PROJECT_ID=<proj> bash scripts/hpc_scripts/adsorbml/setup_perun_uma_env.sh --hf-token hf_xxx   # -> ~/envs/uma, /project/<proj>/hf_cache
# from the login node: submit each step (step 2 requires step 1's manifest to exist first)
STEP=1 ACCOUNT=<proj> UMA_PYTHON=$HOME/envs/uma/bin/python [INCLUDE="Mo2N_*"] bash scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh
STEP=2 ACCOUNT=<proj> UMA_PYTHON=$HOME/envs/uma/bin/python                    bash scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh
python scripts/adsorbml/3-extract_rank.py                                     # rank locally (CPU-only)
```
- `setup_perun_uma_env.sh`: loads a site Python module, builds `~/envs/uma` with the aarch64
  CUDA `torch==2.8.0+cu129` wheel + fairchem (`PYTHON_MODULE=` knob if the default name is wrong).
- `submit_perun_adsorbml.sh` + `perun_adsorbml_worker.sh`: one GPU job per step
  (auto-parallel across the node's GPUs; `GRES=gpu:4` for a full node). Outputs go to
  `$ADSORBML_DATA_ROOT` as usual.
- `uma-m-1p1` is a **gated** HF model with no repo-side auth handling — `setup_perun_uma_env.sh`
  warms the cache once (needs `HF_TOKEN` + license accepted) so jobs run `HF_HUB_OFFLINE=1`.
- Arch/env/partition details + caveats (incl. why native, not a container): the `/perun-hpc` skill.

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
  the LDA→PBE→RPBE/PW switch, the silent no-op vacancy/dopant + interface-overlap generator
  bugs, wrong base polymorphs (MoP/Mo₂N/MoS₂), and that the old gpaw ΔG_H CSV is not
  trustworthy ground truth.

## Status / direction (from README, 2026-02-11)

MoS₂/MoSe₂ basal planes are inert → focus on **edges and defects**. **Mo₂N** is the
strongest candidate; refine with dopants/vacancies. Use AdsorbML/UMA to shortlist,
GPAW to compute, VASP only for final validation of top candidates.
