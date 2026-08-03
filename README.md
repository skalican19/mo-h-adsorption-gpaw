# Mo H Adsorption Workflow

This repository contains the scripts, inputs, outputs, and documentation for computing hydrogen adsorption energies ($\Delta G_H$) on Mo-based compounds (MoS₂, MoSe₂, MoP, Mo₂N) using GPAW.

## Structure

- scripts/ — runnable scripts (GPAW + structure preparation)
- data/inputs/ — POSCAR inputs
- data/outputs/ — results and GPAW calculation logs
- docs/ — methodology and reports

## Quick Start

1. Create POSCAR inputs
   - Run scripts/generate_structures.py
   - Generates basal slabs, Mo/Chalcogen edge ribbons, S/Se vacancies,
     Mo2N vacancies/dopants, and Ni/Mo2N interface models
2. Run GPAW calculations
   - Run scripts/gpaw_h_adsorption.py
   - Auto-discovers all POSCAR directories under data/inputs/VASP_inputs
   - DEVANA: prefer one structure per SLURM task via scripts/submit_devana_gpaw_array.sh
   - Desktop nodes (node1/node2/node3): run directly via scripts/run_desktop_machine.sh
3. Results
   - data/outputs/gpaw_h_adsorption_results.csv

## Main Outputs

- data/outputs/gpaw_h_adsorption_results.csv
- data/outputs/gpaw_h_adsorption_results.json

## Notes

- GPAW must be installed and accessible in your environment.
- Calculations are configured for slab models with vacuum spacing.
- Noble metal dopant shortlist is in dopant_shortlist.json.
- DEVANA usage:
   - Generate and submit an array with ACCOUNT=myproject bash scripts/submit_devana_gpaw_array.sh
   - Each array task runs exactly one structure through scripts/devana_gpaw_array_worker.sh
- Desktop usage (no SLURM):
   - Run MACHINE=node1 bash scripts/run_desktop_machine.sh
   - Run MACHINE=node2 bash scripts/run_desktop_machine.sh
   - Run MACHINE=node3 bash scripts/run_desktop_machine.sh
   - Tune local concurrency with CORES_PER_CALC and optional WORKERS

## AdsorbML ML screening (Perun GPU)

Fast UMA-based H* screening to shortlist candidates before GPAW. On the Perun HPC GPU
nodes (aarch64/GH200) it runs from an NVIDIA NGC container:

1. Build the env once, on a GPU node (`srun --partition=gpu_short --gres=gpu:1 --pty bash`):
   - `PROJECT_ID=<proj> bash scripts/hpc_scripts/adsorbml/setup_perun_uma_env.sh --hf-token hf_xxx`
   - Builds the container + fairchem venv and caches the gated `uma-m-1p1` weights to `/project/<proj>/hf_cache`.
2. Submit the steps from the login node (step 2 needs step 1's manifest first):
   - `STEP=1 ACCOUNT=<proj> SIF_PATH=/project/<proj>/containers/pytorch-ngc.dir [INCLUDE="Mo2N_*"] bash scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh`
   - `STEP=2 ACCOUNT=<proj> SIF_PATH=/project/<proj>/containers/pytorch-ngc.dir bash scripts/hpc_scripts/adsorbml/submit_perun_adsorbml.sh`
3. Rank locally (CPU-only): `python scripts/adsorbml/3-extract_rank.py`

See the `/perun-hpc` skill (or CLAUDE.md) for arch/env details and caveats.

## Updated Direction (2026-02-11)

- MoS2/MoSe2 basal planes are inert; next modeling should target edge sites and defects.
- Mo2N remains the strongest candidate; refine with defects/dopants.
- Use OCx24 to shortlist noble metal dopants for Ni/Mo composite interfaces.
- Use VASP only for final validation of top candidates.
