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

## Updated Direction (2026-02-11)

- MoS2/MoSe2 basal planes are inert; next modeling should target edge sites and defects.
- Mo2N remains the strongest candidate; refine with defects/dopants.
- Use OCx24 to shortlist noble metal dopants for Ni/Mo composite interfaces.
- Use VASP only for final validation of top candidates.
