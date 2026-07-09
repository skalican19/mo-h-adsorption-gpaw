# Add Mo H adsorption GPAW workflow

## Summary
- Adds a self‑contained GPAW workflow repo with scripts, inputs, and outputs.
- Includes structure generation and adsorption calculations.
- Provides data outputs (CSV/JSON) for ML and analysis.

## Key changes
- New scripts in scripts/:
  - generate_structures.py
  - gpaw_h_adsorption.py
  - compute_h_adsorption.py (VASP optional)
  - parse_vasp_results.py
- Data organized under data/inputs and data/outputs.
- Added README.md, requirements.txt, and .gitignore.

## Data
- Includes small input/output data (<1 MB total).
- Heavy GPAW logs excluded via .gitignore.

## Notes
- Docs folder is ignored in git by request.
- Repo is self‑contained; does not depend on parent ocx code.
