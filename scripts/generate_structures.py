"""
Generate POSCAR files for Mo compounds
Creates surface slabs using literature lattice parameters and ASE
"""

import argparse
import fnmatch
import sys
from pathlib import Path
from ase import Atoms
from ase.io import write
from ase.constraints import FixAtoms
from ase.build import bulk as ase_bulk, surface, mx2, make_supercell
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
import numpy as np

from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder


# ── Validation helpers ───────────────────────────────────────────

def _coordination_counts(atoms, symbol_a, symbol_b, cutoff):
    """Per-image-aware count of symbol_b neighbors within cutoff for each symbol_a atom."""
    symbols = atoms.get_chemical_symbols()
    a_indices = [i for i, s in enumerate(symbols) if s == symbol_a]
    counts = {i: 0 for i in a_indices}
    i_arr, j_arr = neighbor_list('ij', atoms, cutoff=cutoff)
    for a_idx, b_idx in zip(i_arr, j_arr):
        if symbols[a_idx] == symbol_a and symbols[b_idx] == symbol_b:
            counts[a_idx] += 1
    return list(counts.values())


def _assert_coordination(atoms, symbol_a, symbol_b, cutoff, expected, label):
    """Raise if every symbol_a atom doesn't have exactly `expected` symbol_b neighbors."""
    counts = _coordination_counts(atoms, symbol_a, symbol_b, cutoff)
    if not counts or any(c != expected for c in counts):
        raise ValueError(
            f"{label}: expected every {symbol_a} atom to have {expected} {symbol_b} "
            f"neighbor(s) within {cutoff} A, got {counts or '[]'} (empty = no {symbol_a} atoms found)"
        )


def _min_covalent_radius_ratio(atoms):
    """Min (periodic-image-aware pair distance) / (0.6 * sum of covalent radii).

    < 1.0 means some pair of distinct atoms sits closer than a physically
    plausible bond -- i.e. an unphysical overlap, not just a short contact.
    """
    if len(atoms) < 2:
        return np.inf
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    radii = covalent_radii[atoms.get_atomic_numbers()]
    threshold = 0.6 * (radii[:, None] + radii[None, :])
    return float(np.min(d / threshold))


def _z_layers(z_values, tol=0.5):
    """Collapse a 1-D array of z coordinates into sorted layer centroids."""
    zs = np.sort(np.asarray(z_values))
    layers = [[zs[0]]]
    for z in zs[1:]:
        if z - layers[-1][-1] < tol:
            layers[-1].append(z)
        else:
            layers.append([z])
    return np.array([np.mean(l) for l in layers])


# fcc interlayer spacing d_hkl (Å) for a=3.52 Ni, per facet actually used.
# (100) stacks every a/2; (111) every a/sqrt(3). A degenerate primitive-cell
# Ni bulk made both Miller strings cut the same {111} planes -- this table lets
# the substrate assert catch that regression by measuring the real spacing.
_NI_FACET_SPACING = {(1, 0, 0): 3.52 / 2, (1, 1, 1): 3.52 / np.sqrt(3)}


def _assert_substrate_facet(interface, symbol, miller, label):
    """Raise if the substrate sublattice's interlayer spacing != the requested facet."""
    expected = _NI_FACET_SPACING.get(tuple(miller))
    if expected is None:
        return  # facet not tabulated; skip rather than guess
    z_sub = [a.position[2] for a in interface if a.symbol == symbol]
    layers = _z_layers(z_sub)
    if len(layers) < 2:
        raise ValueError(f"{label}: <2 {symbol} layers, cannot verify {miller} facet")
    spacing = float(np.median(np.diff(layers)))
    if abs(spacing - expected) > 0.15:
        raise ValueError(
            f"{label}: {symbol} interlayer spacing {spacing:.2f} A does not match "
            f"the requested {miller} facet (expected {expected:.2f} A) -- likely a "
            f"primitive-cell Miller degeneracy (use cubic=True for the fcc bulk)"
        )


def _assert_film_integrity(interface, substrate_symbol, max_internal_gap, label):
    """Raise if the film sublattice (non-substrate atoms) is split by a vacuum gap.

    Catches the sliced-monolayer artifact where an interface builder cuts a 2-D
    sheet mid-slab, leaving orphan atomic planes separated by >max_internal_gap.
    """
    z_film = [a.position[2] for a in interface if a.symbol != substrate_symbol]
    if not z_film:
        raise ValueError(f"{label}: no film atoms found")
    layers = _z_layers(z_film)
    gaps = np.diff(layers)
    if len(gaps) and gaps.max() > max_internal_gap:
        raise ValueError(
            f"{label}: film is split by a {gaps.max():.1f} A internal vacuum gap "
            f"(> {max_internal_gap} A) -- sheet was sliced, not kept intact"
        )


# ── Base bulk structures ─────────────────────────────────────────

def create_tmd_monolayer(formula, a, thickness):
    """Single 2H-MX2 monolayer in an ORTHOGONAL (rectangular) cell (2 f.u.).

    Used for edge nanoribbons: the P6_3/mmc bulk has two S-Mo-S sheets per cell,
    so repeating it made "edge" ribbons accidental bilayer rods. A true edge
    model is one monolayer wide in z. The rectangular cell (a x a*sqrt(3)) has
    orthogonal axes so a clean finite-in-x / periodic-in-y ribbon can be cut.
    """
    hexol = mx2(formula=formula, kind='2H', a=a, thickness=thickness,
                size=(1, 1, 1), vacuum=7.5)
    ortho = make_supercell(hexol, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])
    ortho.set_pbc([True, True, False])
    return ortho


def create_mos2_bulk():
    """2H-MoS2 bulk, P6_3/mmc (#194), two S-Mo-S layers per cell (mp-1018809)."""
    a, c = 3.160, 12.295  # Å
    lattice = Lattice.hexagonal(a, c)
    struct = Structure.from_spacegroup(
        "P6_3/mmc", lattice, ["Mo", "S"], [[1 / 3, 2 / 3, 1 / 4], [1 / 3, 2 / 3, 0.621]]
    )
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.set_pbc([True, True, True])
    _assert_coordination(atoms, "Mo", "S", cutoff=2.6, expected=6, label="MoS2 bulk")
    return atoms


def create_mose2_bulk():
    """2H-MoSe2 bulk, P6_3/mmc (#194), two Se-Mo-Se layers per cell."""
    a, c = 3.289, 12.929  # Å
    lattice = Lattice.hexagonal(a, c)
    struct = Structure.from_spacegroup(
        "P6_3/mmc", lattice, ["Mo", "Se"], [[1 / 3, 2 / 3, 1 / 4], [1 / 3, 2 / 3, 0.620]]
    )
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.set_pbc([True, True, True])
    _assert_coordination(atoms, "Mo", "Se", cutoff=2.8, expected=6, label="MoSe2 bulk")
    return atoms


def create_mop_bulk():
    """WC-type MoP bulk, hexagonal P-6m2 (#187), mp-219."""
    a, c = 3.23, 3.21  # Å
    lattice = Lattice.hexagonal(a, c)
    struct = Structure.from_spacegroup(
        "P-6m2", lattice, ["P", "Mo"], [[0, 0, 0], [1 / 3, 2 / 3, 0.5]]
    )
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.set_pbc([True, True, True])
    _assert_coordination(atoms, "Mo", "P", cutoff=2.7, expected=6, label="MoP bulk")
    return atoms


def create_mo2n_bulk():
    """beta-Mo2N bulk, anti-anatase I4_1/amd (#141), matching mp-27953.

    Explicit fractional coordinates (an fcc Mo lattice with N filling one
    octahedral site per (001) layer, rotating 90 deg layer-to-layer). Built
    directly rather than via a from_spacegroup Wyckoff lookup because the
    I4_1/amd origin-choice/Wyckoff-label combination for this ordering wasn't
    reproduced cleanly with pymatgen's space-group generators; coordination
    is asserted below instead.
    """
    a, c = 4.20, 8.00  # Å
    cell = np.diag([a, a, c])

    mo_frac = [
        (0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.25), (0.0, 0.5, 0.25),
    ]
    mo_frac += [(x, y, z + 0.5) for (x, y, z) in mo_frac]

    n_frac = [
        (0.5, 0.0, 0.0), (0.5, 0.5, 0.25), (0.0, 0.5, 0.5), (0.0, 0.0, 0.75),
    ]

    symbols = ["Mo"] * len(mo_frac) + ["N"] * len(n_frac)
    positions = np.array(mo_frac + n_frac) % 1.0

    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    _assert_coordination(atoms, "N", "Mo", cutoff=2.3, expected=6, label="Mo2N bulk (N-Mo)")
    _assert_coordination(atoms, "Mo", "N", cutoff=2.3, expected=3, label="Mo2N bulk (Mo-N)")
    return atoms


def create_mo2c_bulk():
    """beta-Mo2C bulk, orthorhombic Pbcn (#60), xi-Fe2N-type (mp / ICSD, exp).

    12 atoms/cell (Mo8 C4): Mo on 8d, C on 4c. The previous 6-atom hand-typed
    cell was HALF density (~4.6 vs ~9.2 g/cc) and reduced to P2_1, not Pbcn --
    it was not beta-Mo2C at all. Coordinates from Christensen (1977) /
    arXiv:2201.12706 Table 1; from_spacegroup expands the Wyckoff orbits so the
    full 12-atom cell is generated correctly.
    """
    a, b, c = 4.725, 6.022, 5.195  # Å, experimental
    lattice = Lattice.orthorhombic(a, b, c)
    struct = Structure.from_spacegroup(
        "Pbcn", lattice, ["Mo", "C"],
        [[0.250, 0.125, 0.083], [0.500, 0.375, 0.250]],
    )
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.set_pbc([True, True, True])
    # Each C sits in an octahedron of 6 Mo; each Mo has 3 near-planar C.
    _assert_coordination(atoms, "C", "Mo", cutoff=2.4, expected=6, label="Mo2C bulk (C-Mo)")
    _assert_coordination(atoms, "Mo", "C", cutoff=2.4, expected=3, label="Mo2C bulk (Mo-C)")
    return atoms


def create_mob_bulk():
    """alpha-MoB bulk, tetragonal I4_1/amd (#141), Bg / CrB-type.

    Kiessling (1947), COD 9008953 / ICSD 24280: Mo and B both on 8e,
    z(Mo)=0.197, z(B)=0.352, a=3.105, c=16.97. Boron forms zigzag chains
    (each B has 2 in-chain B neighbours ~1.74 Å); each B sits in a
    trigonal-prismatic Mo cage. The previous hand-typed "8e" list did not lie
    on a valid I4_1/amd orbit and produced 0.76 Å Mo-B overlaps (detected SG
    Cmmm) -- from_spacegroup generates the correct orbit instead.
    """
    a, c = 3.105, 16.97  # Å, experimental
    lattice = Lattice.tetragonal(a, c)
    struct = Structure.from_spacegroup(
        "I4_1/amd", lattice, ["Mo", "B"], [[0, 0, 0.197], [0, 0, 0.352]]
    )
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.set_pbc([True, True, True])
    _assert_coordination(atoms, "B", "B", cutoff=2.0, expected=2, label="MoB bulk (B-B chain)")
    _assert_coordination(atoms, "B", "Mo", cutoff=2.7, expected=7, label="MoB bulk (B-Mo cage)")
    return atoms


def create_ti3c2_bulk():
    """Ti3C2O2 MXene as a COMPACT vdW-stacked hexagonal cell (O-terminated).

    Seven close-packed sub-layers O-Ti-C-Ti-C-Ti-O (~1.0 Å apart, ~6 Å thick)
    plus a ~3 Å van-der-Waals gap, so c ≈ 9 Å is a real periodic stack -- NOT a
    monolayer floating in 20 Å of vacuum. The old vacuum-cell version made
    SlabGenerator/CoherentInterfaceBuilder treat the whole 20 Å as bulk and
    slice the sheet mid-monolayer; a compact cell puts the (001) cut in the gap
    and yields one intact sheet with film_thickness=1.
    """
    a = 3.071          # Å, in-plane
    sub_dz = 1.0       # Å between close-packed sub-layers
    gap = 3.0          # Å van-der-Waals gap between stacked sheets
    thickness = 6 * sub_dz            # O-to-O sheet thickness
    c = thickness + gap

    cell = np.array([
        [a, 0, 0],
        [-a / 2, a * np.sqrt(3) / 2, 0],
        [0, 0, c],
    ])

    z0 = 0.0                           # sheet contiguous from the origin; gap sits at the top
    #                                    of the cell (not straddling the boundary), so a
    #                                    film_thickness=1 cut keeps all 7 sub-layers together
    symbols = ['O', 'Ti', 'C', 'Ti', 'C', 'Ti', 'O']
    xy = [(1 / 3, 2 / 3), (2 / 3, 1 / 3), (0.0, 0.0), (1 / 3, 2 / 3),
          (2 / 3, 1 / 3), (0.0, 0.0), (1 / 3, 2 / 3)]
    positions = np.array([[x, y, (z0 + i * sub_dz) / c] for i, (x, y) in enumerate(xy)])

    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_graphene_sheet(size=(4, 4, 1), vacuum=10):
    """Create a graphene sheet slab."""
    a = 2.46  # Å, graphene lattice constant

    cell = np.array([
        [a, 0, 0],
        [-a / 2, a * np.sqrt(3) / 2, 0],
        [0, 0, 20.0],
    ])

    symbols = ['C', 'C']
    positions = np.array([
        [0.0, 0.0, 0.5],
        [1 / 3, 2 / 3, 0.5],
    ])

    bulk = Atoms(symbols, cell=cell, pbc=[True, True, True])
    bulk.set_scaled_positions(positions)

    sheet = bulk.repeat(size)
    sheet.set_pbc([True, True, False])
    sheet.center(vacuum=vacuum, axis=2)
    return sheet


def create_n_doped_graphene(size=(4, 4, 1), vacuum=10):
    """Create N-doped graphene (one C replaced by N)."""
    sheet = create_graphene_sheet(size=size, vacuum=vacuum)
    # Replace the C atom closest to center with N
    positions = sheet.get_positions()
    center_xy = np.mean(positions[:, :2], axis=0)
    c_indices = [i for i, atom in enumerate(sheet) if atom.symbol == 'C']
    distances = [np.linalg.norm(positions[i, :2] - center_xy) for i in c_indices]
    replace_idx = c_indices[int(np.argmin(distances))]
    sheet[replace_idx].symbol = 'N'
    return sheet


def _apply_constraints(slab):
    """Freeze bottom half of atoms for stability."""
    z_positions = slab.get_positions()[:, 2]
    z_min = np.min(z_positions)
    z_max = np.max(z_positions)
    z_mid = (z_min + z_max) / 2
    fixed_indices = [i for i in range(len(slab)) if slab[i].z < z_mid]
    slab.set_constraint(FixAtoms(indices=fixed_indices))


def _freeze_substrate_bottom_half(interface, substrate_symbol):
    """Freeze the bottom half of the substrate sublattice only (interface slabs)."""
    indices_sub = [i for i, atom in enumerate(interface) if atom.symbol == substrate_symbol]
    if not indices_sub:
        return
    z_sub = interface.get_positions()[indices_sub, 2]
    z_mid = (np.min(z_sub) + np.max(z_sub)) / 2
    fixed = [i for i in indices_sub if interface[i].z < z_mid]
    interface.set_constraint(FixAtoms(indices=fixed))


def _parse_miller(miller):
    """Convert a string like '(111)' to a Miller index tuple."""
    digits = miller.strip().replace("(", "").replace(")", "")
    if len(digits) != 3 or not digits.isdigit():
        raise ValueError(f"Unsupported Miller index format: {miller}")
    return tuple(int(c) for c in digits)


def create_slab(bulk_atoms, miller="(100)", size=(2, 2, 4), vacuum=8):
    """Create a surface slab for a requested Miller index."""
    indices = _parse_miller(miller)

    # Build an oriented slab first, then expand in-plane for supercell-like coverage.
    slab = surface(bulk_atoms, indices, layers=size[2], vacuum=vacuum, periodic=True)
    slab = slab.repeat((size[0], size[1], 1))
    slab.set_pbc([True, True, False])
    slab.center(vacuum=vacuum, axis=2)
    _apply_constraints(slab)
    return slab


def create_edge_ribbon(bulk_atoms, width=6, length=2, vacuum=8, edge_type="Mo",
                       layers_z=2, edge_depth=1.7, keep_fraction=0.0):
    """Create an edge ribbon: finite in x (two edges), periodic in y, vacuum in x+z.

    The removal species is derived from the actual composition (not a hard-coded
    {S,Se,P} set, which made Mo2C/MoB "_edge_Mo" a silent no-op):
        "Mo" -> remove anion atoms in the edge region, exposing a Mo edge;
                `keep_fraction` of those edge anions are retained instead of
                removed (0.5 -> ~50%-S-covered Mo edge, the HER-active model).
        "X"  -> remove Mo in the edge region, exposing an anion edge.

    Raises if the removal set is empty (no silent no-op). `layers_z` >= 2 makes
    the 3-D carbide/nitride/boride "edges" a genuine nanorod rather than a
    single-cell sliver; TMD monolayer ribbons pass layers_z=1.

    NOTE: `keep_fraction` thins the edge anions to an approximate coverage by a
    deterministic every-other rule; it is a reasonable partial-coverage edge
    approximation, not a rigorously reconstructed literature edge (exact
    monomer/dimer arrangement is not claimed).
    """
    metal = "Mo"
    ribbon = bulk_atoms.repeat((width, length, layers_z))
    ribbon.center(vacuum=vacuum, axis=0)
    ribbon.center(vacuum=vacuum, axis=2)
    ribbon.set_pbc([False, True, False])

    symbols = ribbon.get_chemical_symbols()
    anions = sorted({s for s in symbols if s != metal})
    if edge_type == "Mo":
        remove_species = set(anions)
    else:
        if not anions:
            raise ValueError(f"edge_type={edge_type}: no anion species to expose")
        remove_species = {metal}

    pos = ribbon.get_positions()
    x = pos[:, 0]
    x_min, x_max = x.min(), x.max()
    edge_mask = (x - x_min < edge_depth) | (x_max - x < edge_depth)
    candidates = [i for i, s in enumerate(symbols) if edge_mask[i] and s in remove_species]
    if not candidates:
        raise ValueError(
            f"{edge_type}-edge: no {remove_species} atoms within {edge_depth} A of "
            f"either x-edge to remove; refusing a silent no-op"
        )

    # Retain keep_fraction of the edge anions (Mo-edge coverage control), chosen
    # deterministically by (x, y) so reruns are byte-identical.
    if keep_fraction > 0.0:
        ordered = sorted(candidates, key=lambda i: (round(x[i], 3), round(pos[i, 1], 3)))
        step = max(1, round(1.0 / keep_fraction))
        keep = set(ordered[::step])
        remove = [i for i in candidates if i not in keep]
    else:
        remove = candidates
    if not remove:
        raise ValueError(f"{edge_type}-edge: keep_fraction={keep_fraction} left nothing to remove")

    del ribbon[remove]
    _apply_constraints(ribbon)
    return ribbon


def create_vacancy_slab(slab, vacancy_symbol, tol=0.5):
    """Remove one top-layer atom (of its own species' top layer) to create a vacancy."""
    positions = slab.get_positions()
    z_positions = positions[:, 2]

    target_idx = [i for i, atom in enumerate(slab) if atom.symbol == vacancy_symbol]
    if not target_idx:
        raise ValueError(f"No {vacancy_symbol} atoms in slab; cannot create a vacancy")

    z_top = np.max(z_positions[target_idx])
    candidates = [i for i in target_idx if (z_top - z_positions[i]) < tol]
    if not candidates:
        raise ValueError(
            f"No {vacancy_symbol} atom found within {tol} A of its own top layer "
            f"(z_top={z_top:.2f}); refusing to silently no-op"
        )

    # Remove the candidate closest to xy center (deterministic pick)
    center_xy = np.mean(positions[:, :2], axis=0)
    distances = [np.linalg.norm(positions[i, :2] - center_xy) for i in candidates]
    remove_index = candidates[int(np.argmin(distances))]
    del slab[remove_index]
    return slab


def create_multi_vacancy_slab(slab, vacancy_symbol, count=2, tol=0.5):
    """Remove `count` mutually adjacent top-layer atoms to form a real vacancy cluster."""
    positions = slab.get_positions()
    z_positions = positions[:, 2]

    target_idx = [i for i, atom in enumerate(slab) if atom.symbol == vacancy_symbol]
    if len(target_idx) < count:
        raise ValueError(
            f"Only {len(target_idx)} {vacancy_symbol} atom(s) in slab; cannot remove {count}"
        )

    z_top = np.max(z_positions[target_idx])
    surface_candidates = [i for i in target_idx if (z_top - z_positions[i]) < tol]
    if not surface_candidates:
        raise ValueError(
            f"No {vacancy_symbol} atom found within {tol} A of its own top layer "
            f"(z_top={z_top:.2f}); refusing to silently no-op"
        )

    center_xy = np.mean(positions[:, :2], axis=0)
    seed = min(surface_candidates, key=lambda i: np.linalg.norm(positions[i, :2] - center_xy))

    dmat = slab.get_all_distances(mic=True)
    same_species_dists = [
        dmat[target_idx[a], target_idx[b]]
        for a in range(len(target_idx)) for b in range(a + 1, len(target_idx))
    ]
    nn_dist = min(same_species_dists) if same_species_dists else 0.0
    cluster_cutoff = nn_dist * 1.5

    cluster = [seed]
    remaining = [i for i in target_idx if i != seed]
    while len(cluster) < count and remaining:
        next_idx = min(remaining, key=lambda r: min(dmat[c, r] for c in cluster))
        cluster.append(next_idx)
        remaining.remove(next_idx)

    for a in cluster:
        if not any(dmat[a, b] < cluster_cutoff for b in cluster if b != a):
            raise ValueError(
                f"{vacancy_symbol} vacancy cluster of size {count} is not mutually contiguous "
                f"(nearest-neighbor distance {nn_dist:.2f} A, cutoff {cluster_cutoff:.2f} A)"
            )

    for idx in sorted(cluster, reverse=True):
        del slab[idx]
    return slab


def create_substitution_slab(slab, target_symbol, dopant_symbol, tol=0.5):
    """Substitute one top-layer atom (of the target species' own top layer) with a dopant."""
    positions = slab.get_positions()
    z_positions = positions[:, 2]

    target_idx = [i for i, atom in enumerate(slab) if atom.symbol == target_symbol]
    if not target_idx:
        raise ValueError(f"No {target_symbol} atoms in slab; cannot place {dopant_symbol} dopant")

    z_top = np.max(z_positions[target_idx])
    candidates = [i for i in target_idx if (z_top - z_positions[i]) < tol]
    if not candidates:
        raise ValueError(
            f"No {target_symbol} atom found within {tol} A of its own top layer "
            f"(z_top={z_top:.2f}); facet may be {target_symbol}-poor at the surface, "
            f"refusing to dope a buried atom"
        )

    center_xy = np.mean(positions[:, :2], axis=0)
    distances = [np.linalg.norm(positions[i, :2] - center_xy) for i in candidates]
    replace_index = candidates[int(np.argmin(distances))]
    slab[replace_index].symbol = dopant_symbol
    return slab


def add_cluster_on_surface(slab, element, n_atoms=2, height=1.8, spacing=2.4):
    """Add a small cluster (2 or 4 atoms) above the top surface."""
    positions = slab.get_positions()
    z_max = np.max(positions[:, 2])
    center_xy = np.mean(positions[:, :2], axis=0)

    if n_atoms == 2:
        offsets = [(-spacing / 2, 0.0), (spacing / 2, 0.0)]
    elif n_atoms == 4:
        offsets = [
            (-spacing / 2, -spacing / 2),
            (-spacing / 2, spacing / 2),
            (spacing / 2, -spacing / 2),
            (spacing / 2, spacing / 2),
        ]
    else:
        offsets = [(0.0, 0.0)]

    for dx, dy in offsets:
        slab += Atoms(element, positions=[[center_xy[0] + dx, center_xy[1] + dy, z_max + height]])
    return slab


def _build_zsl_interface(substrate_atoms, film_atoms, substrate_miller, film_miller,
                          separation=2.2, vacuum=15.0, strain_tol=0.02,
                          substrate_thickness=4, film_thickness=2, max_atoms=None):
    """Lattice-match a substrate/film pair with pymatgen ZSL and return a combined ASE slab.

    Replaces the old concatenate-then-set_cell(substrate.cell) approach, which never
    lattice-matched the two in-plane periodicities and wrapped mismatched atoms on
    top of each other (0.22-0.92 A overlaps).

    `max_atoms`, if set, rejects coincidence cells above that size (GPAW
    feasibility); the smallest-strain match that also fits is returned.
    """
    substrate_struct = AseAtomsAdaptor.get_structure(substrate_atoms)
    film_struct = AseAtomsAdaptor.get_structure(film_atoms)

    zsl = ZSLGenerator(max_area_ratio_tol=strain_tol, bidirectional=True)
    builder = CoherentInterfaceBuilder(
        substrate_structure=substrate_struct,
        film_structure=film_struct,
        film_miller=film_miller,
        substrate_miller=substrate_miller,
        zslgen=zsl,
    )

    candidates = []
    for termination in builder.terminations:
        for interface in builder.get_interfaces(
            termination, gap=separation, vacuum_over_film=vacuum,
            film_thickness=film_thickness, substrate_thickness=substrate_thickness,
        ):
            strain = interface.interface_properties['von_mises_strain']
            if strain <= strain_tol:
                candidates.append((strain, interface, termination))
    candidates.sort(key=lambda c: c[0])

    # Low strain (areal/angular lattice match) does not guarantee a physically
    # sane structure -- SlabGenerator's own primitive-cell/supercell reduction
    # can still produce a short in-plane contact within the film or substrate
    # sublattice. Reject those explicitly instead of trusting strain alone.
    too_big = 0
    for strain, interface, termination in candidates:
        atoms = AseAtomsAdaptor.get_atoms(interface)
        atoms.set_pbc([True, True, False])
        if max_atoms is not None and len(atoms) > max_atoms:
            too_big += 1
            continue
        if _min_covalent_radius_ratio(atoms) >= 1.0:
            print(f"[ZSL termination={termination} strain={strain:.3%} natoms={len(atoms)}] ", end="")
            return atoms

    if not candidates:
        raise ValueError(
            f"No ZSL lattice match within {strain_tol:.1%} strain for "
            f"substrate_miller={substrate_miller}, film_miller={film_miller} "
            f"(mismatched lattices; refusing to emit an overlapping structure)"
        )
    if too_big and too_big == len(candidates):
        raise ValueError(
            f"All {len(candidates)} ZSL match(es) within {strain_tol:.1%} strain for "
            f"substrate_miller={substrate_miller}, film_miller={film_miller} exceed the "
            f"{max_atoms}-atom cap (smallest-strain match still too large); "
            f"too big for GPAW"
        )
    raise ValueError(
        f"All {len(candidates)} ZSL match(es) within {strain_tol:.1%} strain for "
        f"substrate_miller={substrate_miller}, film_miller={film_miller} have a "
        f"sub-covalent-radius atom overlap (best strain {candidates[0][0]:.3%}); "
        f"refusing to emit an overlapping structure"
    )


def create_ni_mox_interface(mox_bulk_builder, miller="(111)", separation=2.2, strain_tol=0.02,
                            film_miller=None):
    """Create a Ni/MoX interface slab via ZSL lattice matching (MoX film on Ni substrate).

    `film_miller` defaults to the same facet as the Ni substrate; pass an
    explicit tuple to cut the film on a fixed plane (e.g. the basal (001) of a
    layered vdW crystal) while `miller` still selects only the Ni facet.
    """
    indices = _parse_miller(miller)
    # cubic=True: the primitive fcc cell makes Miller (100) and (111) cut the
    # SAME conventional {111} planes -- a degeneracy that silently gave every
    # "_(100)" interface a Ni(111) substrate. The conventional 4-atom cell
    # interprets Miller indices in the cubic basis as intended.
    ni_bulk = ase_bulk("Ni", "fcc", a=3.52, cubic=True)
    mox_bulk = mox_bulk_builder()
    film_idx = indices if film_miller is None else tuple(film_miller)
    interface = _build_zsl_interface(
        ni_bulk, mox_bulk, substrate_miller=indices, film_miller=film_idx,
        separation=separation, strain_tol=strain_tol,
    )
    _assert_substrate_facet(interface, "Ni", indices, label=f"Ni/MoX {miller}")
    _freeze_substrate_bottom_half(interface, substrate_symbol="Ni")
    return interface


def create_ni_mxene_interface(miller="(111)", separation=2.2, strain_tol=0.03, max_atoms=350):
    """Create a Ni/Ti3C2O2 MXene interface via ZSL lattice matching.

    The MXene film is always cut at its own basal (001) plane (the only
    physically sensible facet of a single 2D sheet), one sheet thick; `miller`
    only selects the Ni substrate facet. Strain tol is relaxed to 3% and a
    350-atom cap applied so a GPAW-feasible coincidence cell can win; if none
    fits, _build_zsl_interface raises and the caller records it UMA-only.
    """
    ni_bulk = ase_bulk("Ni", "fcc", a=3.52, cubic=True)
    mxene_bulk = create_ti3c2_bulk()
    indices = _parse_miller(miller)
    interface = _build_zsl_interface(
        ni_bulk, mxene_bulk, substrate_miller=indices, film_miller=(0, 0, 1),
        separation=separation, strain_tol=strain_tol, film_thickness=1,
        max_atoms=max_atoms,
    )
    _assert_substrate_facet(interface, "Ni", indices, label=f"Ni/MXene {miller}")
    # One intact O-Ti-C-Ti-C-Ti-O sheet: no >3 Å gap within the film sublattice.
    _assert_film_integrity(interface, substrate_symbol="Ni", max_internal_gap=3.0,
                           label=f"Ni/MXene {miller}")
    _freeze_substrate_bottom_half(interface, substrate_symbol="Ni")
    return interface


def create_ni_on_graphene(ni_atoms=4, height=1.8, size=(4, 4, 1), vacuum=10):
    """Create Ni cluster on graphene sheet."""
    sheet = create_graphene_sheet(size=size, vacuum=vacuum)
    sheet = add_cluster_on_surface(sheet, "Ni", n_atoms=ni_atoms, height=height)
    _apply_constraints(sheet)
    return sheet


def create_ni_on_n_doped_graphene(ni_atoms=4, height=1.8, size=(4, 4, 1), vacuum=10):
    """Create Ni cluster on N-doped graphene sheet."""
    sheet = create_n_doped_graphene(size=size, vacuum=vacuum)
    sheet = add_cluster_on_surface(sheet, "Ni", n_atoms=ni_atoms, height=height)
    _apply_constraints(sheet)
    return sheet


def create_graphene_nanoribbon(width=6, length=3, vacuum=8):
    """Create armchair graphene nanoribbon (CNT-like approximation)."""
    a = 2.46
    cell = np.array([
        [a, 0, 0],
        [-a / 2, a * np.sqrt(3) / 2, 0],
        [0, 0, 20.0],
    ])
    symbols = ['C', 'C']
    positions = np.array([
        [0.0, 0.0, 0.5],
        [1 / 3, 2 / 3, 0.5],
    ])
    bulk = Atoms(symbols, cell=cell, pbc=[True, True, True])
    bulk.set_scaled_positions(positions)

    ribbon = bulk.repeat((width, length, 1))
    ribbon.center(vacuum=vacuum, axis=0)
    ribbon.center(vacuum=vacuum, axis=2)
    ribbon.set_pbc([False, True, False])
    _apply_constraints(ribbon)
    return ribbon


def create_interface_with_dopant_generic(mox_bulk_builder, miller, dopant, target_symbol="Mo",
                                         film_miller=None):
    """Create Ni/MoX interface with a single dopant on the exposed MoX top layer.

    Targets the topmost Mo (the exposed catalytic surface; MoX sits on top of
    the buried Ni substrate) rather than Ni. If the facet is anion-terminated
    (no surface Mo), create_substitution_slab raises instead of doping a
    buried atom.
    """
    interface = create_ni_mox_interface(mox_bulk_builder, miller=miller, film_miller=film_miller)
    interface = create_substitution_slab(interface, target_symbol, dopant)
    return interface


def create_interface_with_cluster_generic(mox_bulk_builder, miller, dopant, cluster_size,
                                          film_miller=None):
    """Create Ni/MoX interface with a small dopant cluster on surface."""
    interface = create_ni_mox_interface(mox_bulk_builder, miller=miller, film_miller=film_miller)
    interface = add_cluster_on_surface(interface, dopant, n_atoms=cluster_size)
    return interface


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUTS = REPO_ROOT / "data" / "inputs" / "VASP_inputs"

# Facets per material. Hexagonal/tetragonal materials get basal + physically
# meaningful facets instead of cubic-motivated (100)/(110)/(111); materials not
# listed keep the historical cubic-ish default (Mo2C, MoB - out of scope for
# this pass; ase.build.surface() cuts these Miller strings correctly regardless
# of crystal system, so no parser change is needed for any of them).
DEFAULT_MILLERS = ['(100)', '(110)', '(111)']
FACETS = {
    'MoS2': ['(001)'],
    'MoSe2': ['(001)'],
    'MoP': ['(001)'],
    'Mo2N': ['(001)', '(100)', '(111)', '(112)'],
}

# In-plane supercell for defect/dopant slabs (dilutes the defect vs. its
# periodic images; ~1/9 coverage instead of the old 1/4 from a 2x2 cell).
DEFECT_SIZE = (3, 3, 4)

# 2H-TMD monolayer params for edge ribbons: (in-plane a, S-S vertical thickness) Å.
TMD_PARAMS = {
    'MoS2':  (3.160, 3.19),
    'MoSe2': (3.289, 3.34),
}


def generate_all_structures(include_glob=None, out_dir=None, list_only=False, dry_run=False):
    """Generate all structure files.

    Returns a list of (name, error_message) for any structure that failed to
    build; callers should treat a non-empty list as a hard failure.
    """

    base_dir = Path(out_dir) if out_dir else DATA_INPUTS
    base_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    def write_structure(name, builder_fn):
        _write_structure(base_dir, name, builder_fn, include_glob=include_glob,
                          list_only=list_only, dry_run=dry_run, failures=failures)

    # Map formulas to builder functions
    builders = {
        'MoS2': create_mos2_bulk,
        'MoSe2': create_mose2_bulk,
        'MoP': create_mop_bulk,
        'Mo2N': create_mo2n_bulk,
        'Mo2C': create_mo2c_bulk,
        'MoB': create_mob_bulk,
    }

    # Chalcogenide-like compounds that get vacancy/edge/sheet variants
    chalcogenides = {
        'MoS2': 'S',
        'MoSe2': 'Se',
    }

    # Compounds that get dopant/vacancy treatment (metal sublattice + anion)
    dopant_compounds = {
        'Mo2N': {'metal': 'Mo', 'anion': 'N'},
        'Mo2C': {'metal': 'Mo', 'anion': 'C'},
        'MoB':  {'metal': 'Mo', 'anion': 'B'},
    }

    # Systems that get Ni/MoX interface treatment. `film_miller=None` cuts the
    # film on the same facet as the Ni substrate (fine for the 3-D carbides/
    # nitride/boride); MoS2 is a vdW crystal, so its film is always cut on the
    # basal (001) plane -- cutting through (111)/(100) would sever covalent
    # S-Mo-S bonds and expose a non-physical broken-bond termination.
    interface_systems = {
        'Ni_Mo2N': {'builder': create_mo2n_bulk, 'film_miller': None},
        'Ni_Mo2C': {'builder': create_mo2c_bulk, 'film_miller': None},
        'Ni_MoB':  {'builder': create_mob_bulk,  'film_miller': None},
        'Ni_MoS2': {'builder': create_mos2_bulk, 'film_miller': (0, 0, 1)},
    }

    dopants = ["Pt", "Pd", "Ir", "Ru", "Ag", "Au", "Ni"]
    # Anna's decoration list (subset for interfaces); Ru restored so the
    # headline Ni_Mo2N_interface_(111)_cluster4Ru candidate can be rebuilt.
    decorations = ["Ag", "Au", "Pd", "Pt", "Ir", "Ru"]

    interface_millers = ['(111)', '(100)']

    print("\n" + "="*60)
    print("Generating Structure Files for GPAW Calculations")
    print("="*60)

    # ── Part 1: Basic slabs for all compounds ────────────────────
    for formula, builder in builders.items():
        print(f"\n{formula}:")

        try:
            bulk = builder()
            print(f"  Bulk structure: {len(bulk)} atoms/cell")
        except Exception as e:
            print(f"  ✗ Failed to create {formula}: {e}")
            failures.append((formula, str(e)))
            continue

        for miller in FACETS.get(formula, DEFAULT_MILLERS):
            write_structure(f"{formula}_{miller}",
                lambda m=miller, b=bulk: create_slab(b.copy(), miller=m))

    # ── Part 2: Chalcogenide variants (vacancies, edges, sheets) ─
    for formula, vac_sym in chalcogenides.items():
        print(f"\n{formula} variants:")
        bulk = builders[formula]()
        facets = FACETS[formula]
        print(f"    [defect supercell] {DEFECT_SIZE[0]}x{DEFECT_SIZE[1]} in-plane -> "
              f"~{100 / (DEFECT_SIZE[0] * DEFECT_SIZE[1]):.1f}% coverage")

        for miller in facets:
            # Single vacancy
            write_structure(f"{formula}_{miller}_vac{vac_sym}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, size=DEFECT_SIZE), vac_sym))
            # Double vacancy
            write_structure(f"{formula}_{miller}_vac2{vac_sym}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, size=DEFECT_SIZE), vac_sym, count=2))
            # Nanosheet
            write_structure(f"{formula}_{miller}_sheet",
                lambda m=miller: create_slab(bulk.copy(), miller=m, size=(4, 4, 4), vacuum=10))

        # Edge ribbons cut from a single MONOLAYER (layers_z=1), not the 2-sheet
        # bulk -- the old ribbons were accidental bilayer rods. Mo-edge and
        # X(=S/Se)-edge are the two zigzag terminations; the Mo-edge is the
        # literature HER-active site.
        tmd_a, tmd_t = TMD_PARAMS[formula]
        mono = create_tmd_monolayer(formula, a=tmd_a, thickness=tmd_t)
        # Mo-edge kept at ~50% anion coverage (HER-active reconstruction);
        # anion-edge is the bare chalcogen termination.
        for edge_type, label, kf in [("Mo", "Mo", 0.5), ("X", vac_sym, 0.0)]:
            write_structure(f"{formula}_edge_{label}",
                lambda et=edge_type, m=mono, k=kf: create_edge_ribbon(
                    m.copy(), edge_type=et, layers_z=1, keep_fraction=k))
            write_structure(f"{formula}_edge_{label}_large",
                lambda et=edge_type, m=mono, k=kf: create_edge_ribbon(
                    m.copy(), width=10, length=3, edge_type=et, layers_z=1, keep_fraction=k))

    # ── Part 2.5: MoP basal + prismatic edge ─────────────────────
    print("\nMoP edges:")
    mop_bulk = builders['MoP']()
    for edge_type, label in [("Mo", "Mo"), ("X", "P")]:
        write_structure(f"MoP_edge_{label}",
            lambda et=edge_type: create_edge_ribbon(mop_bulk.copy(), edge_type=et))
        write_structure(f"MoP_edge_{label}_large",
            lambda et=edge_type: create_edge_ribbon(mop_bulk.copy(), width=10, length=3, edge_type=et))

    # ── Part 3: Dopant/vacancy compounds (Mo2N, Mo2C, MoB) ──────
    for formula, info in dopant_compounds.items():
        print(f"\n{formula} dopants/vacancies:")
        bulk = builders[formula]()
        metal, anion = info['metal'], info['anion']
        facets = FACETS.get(formula, DEFAULT_MILLERS)
        print(f"    [defect supercell] {DEFECT_SIZE[0]}x{DEFECT_SIZE[1]} in-plane -> "
              f"~{100 / (DEFECT_SIZE[0] * DEFECT_SIZE[1]):.1f}% coverage")

        for miller in facets:
            # Anion vacancy
            write_structure(f"{formula}_{miller}_vac{anion}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, size=DEFECT_SIZE), anion))
            # Metal vacancy
            write_structure(f"{formula}_{miller}_vac{metal}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, size=DEFECT_SIZE), metal))
            # Double vacancies
            write_structure(f"{formula}_{miller}_vac2{anion}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, size=DEFECT_SIZE), anion, count=2))
            write_structure(f"{formula}_{miller}_vac2{metal}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, size=DEFECT_SIZE), metal, count=2))
            # Metal-site dopants
            for dopant in dopants:
                write_structure(f"{formula}_{miller}_dop{dopant}",
                    lambda m=miller, d=dopant: create_substitution_slab(
                        create_slab(bulk.copy(), miller=m, size=DEFECT_SIZE), metal, d))

        # Edge ribbons for Mo2C and MoB
        if formula in ('Mo2C', 'MoB'):
            for edge_type, label in [("Mo", "Mo"), ("X", anion)]:
                write_structure(f"{formula}_edge_{label}",
                    lambda et=edge_type: create_edge_ribbon(bulk.copy(), edge_type=et))
                write_structure(f"{formula}_edge_{label}_large",
                    lambda et=edge_type: create_edge_ribbon(bulk.copy(), width=10, length=3, edge_type=et))

            # Nanosheet
            for miller in facets:
                write_structure(f"{formula}_{miller}_sheet",
                    lambda m=miller: create_slab(bulk.copy(), miller=m, size=(4, 4, 4), vacuum=10))

    # ── Part 4: Ni/MoX interfaces ────────────────────────────────
    for sys_name, info in interface_systems.items():
        print(f"\n{sys_name} interfaces:")
        mox_builder = info['builder']
        fm = info['film_miller']

        for miller in interface_millers:
            # Pristine interface
            write_structure(f"{sys_name}_interface_{miller}",
                lambda m=miller, b=mox_builder, f=fm: create_ni_mox_interface(b, miller=m, film_miller=f))

            # Single-atom dopants on the exposed MoX top layer
            for dopant in dopants:
                write_structure(f"{sys_name}_interface_{miller}_dop{dopant}",
                    lambda m=miller, b=mox_builder, d=dopant, f=fm:
                        create_interface_with_dopant_generic(b, m, d, film_miller=f))

            # Noble metal clusters (2 and 4 atoms)
            for dec in decorations:
                for cs in [2, 4]:
                    write_structure(f"{sys_name}_interface_{miller}_cluster{cs}{dec}",
                        lambda m=miller, b=mox_builder, d=dec, s=cs, f=fm:
                            create_interface_with_cluster_generic(b, m, d, s, film_miller=f))

    # ── Part 5: MXene Ti3C2 + Ni ─────────────────────────────────
    print("\nNi/MXene Ti3C2:")

    # Bare MXene slab: only the basal (001) monolayer is physical. Non-basal
    # cuts of a single-sheet vdW cell are ribbon-stack artifacts (40 Å in-plane
    # vacuum), so only (001) with a single sheet (size z=1) is emitted.
    mxene_bulk = create_ti3c2_bulk()
    write_structure("Ti3C2O2_(001)",
        lambda: create_slab(mxene_bulk.copy(), miller="(001)", size=(3, 3, 1), vacuum=10))

    # Ni/MXene interfaces
    for miller in interface_millers:
        write_structure(f"Ni_Ti3C2O2_interface_{miller}",
            lambda m=miller: create_ni_mxene_interface(miller=m))
        # Decorations on Ni/MXene
        for dec in decorations:
            for cs in [2, 4]:
                write_structure(f"Ni_Ti3C2O2_interface_{miller}_cluster{cs}{dec}",
                    lambda m=miller, d=dec, s=cs: add_cluster_on_surface(
                        create_ni_mxene_interface(miller=m), d, n_atoms=s))

    # ── Part 6: Ni on carbon variants ────────────────────────────
    print("\nNi on carbon:")

    # Pristine graphene
    write_structure("graphene_sheet",
        lambda: create_graphene_sheet())

    # N-doped graphene
    write_structure("graphene_N_doped",
        lambda: create_n_doped_graphene())

    # Graphene nanoribbon (CNT approximation)
    write_structure("graphene_nanoribbon",
        lambda: create_graphene_nanoribbon())

    # Ni on graphene
    for n_ni in [2, 4]:
        write_structure(f"Ni{n_ni}_on_graphene",
            lambda n=n_ni: create_ni_on_graphene(ni_atoms=n))
        write_structure(f"Ni{n_ni}_on_graphene_N_doped",
            lambda n=n_ni: create_ni_on_n_doped_graphene(ni_atoms=n))

    # Ni on nanoribbon
    write_structure("Ni4_on_nanoribbon",
        lambda: add_cluster_on_surface(
            create_graphene_nanoribbon(), "Ni", n_atoms=4))

    print("\n" + "="*60)
    if failures:
        print(f"✗ Structure generation finished with {len(failures)} failure(s):")
        for name, err in failures:
            print(f"    {name}: {err}")
    else:
        print("✓ Structure generation complete!")
    print("="*60)

    return failures


def _write_structure(base_dir, name, builder_fn, include_glob=None,
                      list_only=False, dry_run=False, failures=None):
    """Helper: build a structure and write its POSCAR (or validate/list only)."""
    if include_glob and not fnmatch.fnmatch(name, include_glob):
        return
    if list_only:
        print(f"    {name}")
        return

    dir_name = base_dir / name
    poscar_file = dir_name / "POSCAR"
    print(f"    {name}: ", end="", flush=True)
    try:
        slab = builder_fn()
        # Global overlap gate: every emitted structure -- not just interfaces --
        # must be free of sub-covalent-radius contacts. This is the single choke
        # point that would have caught the 0.76 A MoB overlaps at write time.
        ratio = _min_covalent_radius_ratio(slab)
        if ratio < 1.0:
            raise ValueError(
                f"atom overlap: min covalent-radius ratio {ratio:.2f} < 1.0 "
                f"(some pair closer than a physical bond)"
            )
        if dry_run:
            print(f"✓ (dry-run, {len(slab)} atoms, not written)")
            return
        dir_name.mkdir(parents=True, exist_ok=True)
        write(str(poscar_file), slab, format='vasp')
        print(f"✓ ({len(slab)} atoms)")
    except Exception as e:
        print(f"✗ Error: {e}")
        if poscar_file.exists():
            poscar_file.unlink()
        if failures is not None:
            failures.append((name, str(e)))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate POSCAR inputs for Mo compound structures.")
    parser.add_argument("--include", default=None,
        help="fnmatch glob over structure names, e.g. 'Mo2N_*'")
    parser.add_argument("--out-dir", default=None,
        help="Output directory (default: data/inputs/VASP_inputs)")
    parser.add_argument("--list", action="store_true",
        help="List structure names that would be generated, without building them")
    parser.add_argument("--dry-run", action="store_true",
        help="Build and validate each structure, but do not write POSCAR files")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    build_failures = generate_all_structures(
        include_glob=args.include,
        out_dir=args.out_dir,
        list_only=args.list,
        dry_run=args.dry_run,
    )
    if build_failures:
        sys.exit(1)
    if not args.list:
        print("\n✓ Done! All POSCAR files created.")
