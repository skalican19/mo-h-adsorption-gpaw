"""
Audit gate for generated structure inputs.

Walks data/inputs/VASP_inputs/*/POSCAR (or --dir) and reports, per structure:
  - the minimum periodic-image-aware interatomic distance, flagged against a
    per-pair covalent-radius threshold (not a flat cutoff -- a flat 0.7 A
    threshold let real overlaps in Ni_MoS2_interface_(111) pass in the
    2026-07-08/09 audit)
  - the fraction of atoms sitting outside the cell's in-plane box (a signal
    for the old concatenate-then-set_cell interface bug)
  - for every `*_vac*`/`*_dop*` structure, whether it differs (composition or
    coordinates) from its pristine parent -- catches the silent-no-op bug
    class directly, by construction, not just its symptoms

Exit code is non-zero if any violation is found.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from ase.data import covalent_radii
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "inputs" / "VASP_inputs"

# Matches the vac/dop suffixes _write_structure produces, e.g. "_vacS",
# "_vac2N", "_dopPt" -- captures everything up to that suffix as the parent name.
# The trailing element symbol is constrained to Aa-shape (one capital, optional
# one lowercase) so this doesn't false-positive on unrelated names that happen
# to end in the English word "doped", e.g. "graphene_N_doped".
_DEFECT_SUFFIX_RE = re.compile(r"^(?P<parent>.+?)_(?:vac2?[A-Z][a-z]?|dop[A-Z][a-z]?)$")


def min_covalent_radius_ratio(atoms):
    """Min (mic pair distance) / (0.6 * sum of covalent radii); < 1.0 is an overlap."""
    if len(atoms) < 2:
        return np.inf, (None, None)
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    radii = covalent_radii[atoms.get_atomic_numbers()]
    threshold = 0.6 * (radii[:, None] + radii[None, :])
    ratio = d / threshold
    i, j = np.unravel_index(np.argmin(ratio), ratio.shape)
    return float(ratio[i, j]), (atoms[int(i)].symbol, atoms[int(j)].symbol)


def out_of_box_fraction(atoms):
    """Fraction of atoms whose scaled (fractional) in-plane coordinates fall
    outside [0, 1) before wrapping -- a signal for mismatched-lattice gluing.

    Only meaningful on a near-orthogonal in-plane cell (as the old
    fcc111/fcc100-substrate interface builder always produced): on an oblique
    cell -- e.g. the ZSL-matched interfaces this generator now produces --
    atoms legitimately have fractional coordinates outside [0, 1) purely from
    the cell's origin/shape, with no physical overlap at all. Returns
    (fraction, is_meaningful); callers should only treat a nonzero fraction as
    a violation when is_meaningful is True.
    """
    if len(atoms) == 0:
        return 0.0, True
    angles = atoms.cell.angles()
    is_orthogonal = all(abs(a - 90.0) < 2.0 for a in angles)
    scaled = atoms.get_scaled_positions(wrap=False)
    out = (scaled[:, :2] < -1e-6) | (scaled[:, :2] >= 1 + 1e-6)
    return float(np.mean(np.any(out, axis=1))), is_orthogonal


def differs_from_parent(child, parent):
    """True if child's composition or coordinates differ from parent's."""
    if len(child) != len(parent):
        return True
    if sorted(child.get_chemical_symbols()) != sorted(parent.get_chemical_symbols()):
        return True
    if list(child.get_chemical_symbols()) != list(parent.get_chemical_symbols()):
        return True
    return not np.allclose(child.get_positions(), parent.get_positions(), atol=1e-6)


def find_parent_name(name):
    match = _DEFECT_SUFFIX_RE.match(name)
    return match.group("parent") if match else None


def audit(base_dir, overlap_ratio_min=1.0, out_of_box_max=0.0):
    structures = {}
    for poscar in sorted(base_dir.glob("*/POSCAR")):
        name = poscar.parent.name
        try:
            structures[name] = read(poscar, format="vasp")
        except Exception as e:
            print(f"✗ {name}: failed to read POSCAR ({e})")

    violations = []

    print(f"Auditing {len(structures)} structures under {base_dir}\n")

    for name, atoms in structures.items():
        ratio, pair = min_covalent_radius_ratio(atoms)
        oob, oob_meaningful = out_of_box_fraction(atoms)

        problems = []
        if ratio < overlap_ratio_min:
            problems.append(
                f"sub-covalent-radius contact ({pair[0]}-{pair[1]}, ratio={ratio:.2f})"
            )
        if oob_meaningful and oob > out_of_box_max:
            problems.append(f"{oob:.1%} of atoms outside the in-plane cell")

        parent_name = find_parent_name(name)
        if parent_name is not None:
            parent = structures.get(parent_name)
            if parent is None:
                problems.append(f"parent '{parent_name}' not found (cannot check for no-op)")
            elif not differs_from_parent(atoms, parent):
                problems.append(f"identical to pristine parent '{parent_name}' (silent no-op)")

        if problems:
            violations.append((name, problems))
            print(f"✗ {name} ({len(atoms)} atoms): " + "; ".join(problems))

    print(f"\n{len(structures) - len(violations)}/{len(structures)} structures clean.")
    if violations:
        print(f"\n{len(violations)} violation(s):")
        for name, problems in violations:
            for p in problems:
                print(f"  {name}: {p}")
    return violations


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR),
        help="Directory containing <structure_name>/POSCAR subdirectories")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    violations = audit(Path(args.dir))
    sys.exit(1 if violations else 0)
