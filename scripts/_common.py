"""
scripts/_common.py

Shared helpers used by both the standard GPAW pipeline (gpaw_h_adsorption.py)
and the AdsorbML pipeline (adsorbml/1-relax_uma_omat.py): discovering POSCAR
structure directories under data/inputs/VASP_inputs/. Kept dependency-free
(stdlib only) so importing it never drags in gpaw/ase/fairchem.
"""
import fnmatch
from pathlib import Path


def discover_structures(base_dir, include_patterns=None):
    """Discover all POSCAR files under base_dir and return labels.

    Args:
        base_dir: directory containing structure subdirectories
        include_patterns: optional list of glob patterns (e.g. ["Ni_Mo2C_*", "Mo2N_*"]).
            If provided, only directories matching at least one pattern are included.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    items = []
    for entry in sorted(base_path.iterdir()):
        if not entry.is_dir():
            continue
        poscar = entry / "POSCAR"
        if not poscar.exists():
            continue
        # Apply include filter
        if include_patterns:
            if not any(fnmatch.fnmatch(entry.name, pat) for pat in include_patterns):
                continue
        parts = entry.name.split("_", 1)
        formula = parts[0]
        surface = parts[1] if len(parts) > 1 else "unknown"
        items.append((formula, surface, entry))
    return items
