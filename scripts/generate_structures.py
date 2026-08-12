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
from ase.build import bulk as ase_bulk, mx2, make_supercell
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
import numpy as np

from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder

# The correct z-layer grouping already exists in the AdsorbML tagging module, which
# is pure numpy (its ASE imports are lazy; no fairchem, no torch), so reusing it
# adds no dependency and leaves one implementation instead of two. This file is in
# scripts/, so its own directory is what makes `adsorbml` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adsorbml.tagging import layer_groups


# ── Slab sizing convention (OC20) ────────────────────────────────
#
# These are OC20's own numbers, taken from the code that built the dataset UMA's
# `oc20` head was trained on (fairchem/data/oc/core/slab.py):
#
#     SlabGenerator(min_slab_size=7.0, min_vacuum_size=20.0, lll_reduce=False,
#                   center_slab=True, primitive=True, max_normal_search=1)
#     get_slabs(tol=0.3, bonds=None, max_broken_bonds=0, symmetrize=False)
#     tile_atoms(min_ab=8.0)
#
# and stated in the OC20 paper as "a depth of at least 7 Å and a width of at least
# 8 Å" with "a vacuum layer of at least 20 Å". Our DFT settings already match OC20
# (RPBE / PW 350 eV / no spin / fmax 0.03), so its geometry convention is the
# self-consistent choice.
#
# Two distinct notions of "thickness" matter here and conflating them is what made
# the old slabs unreviewable:
#   * MATERIAL thickness = atom span + one interlayer spacing. This is what
#     `min_slab_size` means, and what "at least 7 Å" refers to.
#   * ATOM SPAN = z.max() - z.min(). This is what the placement-wrap invariant
#     below cares about, because the wrap acts on atom positions.
# Mo2N(001) has a 6.00 Å span but 8.00 Å of material (4 planes, 2.00 Å apart), so
# it satisfies OC20 despite the span reading below 7.
MIN_SLAB_THICKNESS = 7.0    # Å, material thickness (OC20 min_slab_size)
MIN_VACUUM = 20.0           # Å (OC20 min_vacuum_size)
MIN_AB = 8.0                # Å, both in-plane vectors (OC20 min_ab)
MAX_ATOMS_TARGET = 250      # OC22's published per-slab ceiling; a warning, not a hard cap
# ZSL coincidence cells are irreducibly larger: an interface is two slabs, and the
# in-plane cell is set by the lattice mismatch rather than by any thickness choice.
# 600 admits every match the current systems produce while still catching a regression
# that blows the cell up -- the pre-fix Ni_MoS2_interface_(100) was 948 atoms. A
# feasibility guard, not a convention.
# ⚠ Headroom is 4 atoms, not the comfortable margin this comment used to claim: the
# largest emitted structure is Ni_MoB_interface_(111)_cluster4* at 596 (the ZSL match
# itself is 592), NOT Ni/Mo2C(111) at 512. A small builder change -- or the ~1 % cell
# change from switching to relaxed lattice constants -- can push a match over the cap,
# after which _build_zsl_interface silently substitutes a higher-strain alternative or
# raises. Re-measure the maximum after anything that moves a lattice constant.
MAX_ATOMS_INTERFACE = 600

# Ni fcc lattice constant, used by every Ni-substrate interface AND by the facet
# spacing table below. It lives here as one constant because the table is a
# regression guard derived FROM it: change the value in the `ase_bulk` calls alone
# and the guard silently starts asserting the wrong spacing.
NI_A = 3.52                 # Å


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
    """Collapse a 1-D array of z coordinates into sorted layer centroids.

    Delegates the grouping to `adsorbml.tagging.layer_groups`, which measures each
    atom against its layer's HIGHEST member rather than against the previously added
    atom. The old local version chained: on a gently sloping sublattice every
    successive atom sat within `tol` of the last one, so the whole slab collapsed into
    a single "layer" -- Mo2C(110) and Mo2C(111) both reported 1 layer instead of 16
    and 21. That merges layers, which SHRINKS the largest measured gap, which is
    exactly the quantity `_assert_film_integrity` uses to detect a sliced film: a
    genuinely severed sheet could have passed.

    `tol` is passed through explicitly (tagging's own default is 0.4 Å) so that
    changing the grouping algorithm did not silently also change the tolerance.
    """
    zs = np.asarray(z_values, dtype=float)
    if zs.size == 0:
        return np.array([])
    centroids = [float(np.mean(zs[group])) for group in layer_groups(zs, tol=tol)]
    return np.array(sorted(centroids))


# fcc interlayer spacing d_hkl (Å) for Ni, per facet actually used.
# (100) stacks every a/2; (111) every a/sqrt(3). A degenerate primitive-cell
# Ni bulk made both Miller strings cut the same {111} planes -- this table lets
# the substrate assert catch that regression by measuring the real spacing.
# Derived from NI_A so it cannot drift out of step with the bulk it checks.
_NI_FACET_SPACING = {(1, 0, 0): NI_A / 2, (1, 1, 1): NI_A / np.sqrt(3)}

# Ni substrate thickness in pymatgen LAYERS, per facet, for the one interface builder
# that must pass layers rather than Å (see create_ni_mxene_interface). Each value is
# the smallest that still clears MIN_SLAB_THICKNESS.
#
# One shared value cannot serve both facets: a "layer" is an oriented-cell repeat, not
# an atomic plane, and the two facets differ in how many planes that is -- on (100) one
# layer is 2 planes (d=1.76 Å), on (111) it is 1 (d=2.03 Å). Measured material thickness
# of the Ni sublattice in the emitted interface:
#     (100): 2 layers -> 4 planes ->  7.04 Å    (was 4 layers -> 8 planes -> 14.08 Å)
#     (111): 4 layers -> 4 planes ->  8.13 Å    (unchanged; 2 layers gives 6.10 Å, thin)
_NI_SUBSTRATE_LAYERS = {(1, 0, 0): 2, (1, 1, 1): 4}


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


# ── Geometry / sizing helpers ────────────────────────────────────

def _atom_span(atoms):
    """z.max() - z.min(). The quantity the placement-wrap invariant is about."""
    z = atoms.get_positions()[:, 2]
    return float(np.max(z) - np.min(z)) if len(atoms) else 0.0


def _material_thickness(atoms, tol=0.5):
    """Atom span plus one interlayer spacing -- how much material the slab represents.

    This is what pymatgen's `min_slab_size` means. A 4-plane slab with 2.0 Å spacing
    spans 6.0 Å of positions but stands for 8.0 Å of bulk, because the periodic repeat
    it was cut from includes the gap above the top plane.
    """
    layers = _z_layers(atoms.get_positions()[:, 2], tol=tol)
    if len(layers) < 2:
        return _atom_span(atoms)
    return _atom_span(atoms) + float(np.median(np.diff(layers)))


def _tile_to_min_ab(atoms, min_ab=MIN_AB):
    """Repeat in a and b until both in-plane vectors are at least `min_ab` Å.

    Mirrors OC20's `tile_atoms` (fairchem/data/oc/core/slab.py). Deliberately a
    threshold on the *cell vectors* rather than a fixed n x n supercell: a fixed
    repeat count makes the physical width depend on the material's lattice
    constant, which is how the old (2,2)/(3,3) sizes ended up spanning anywhere
    from 6.2 to 18 Å.
    """
    la = np.linalg.norm(atoms.cell[0])
    lb = np.linalg.norm(atoms.cell[1])
    na = int(np.ceil(min_ab / la)) if la > 0 else 1
    nb = int(np.ceil(min_ab / lb)) if lb > 0 else 1
    return atoms.repeat((max(na, 1), max(nb, 1), 1))


def _flip_slab_z(slab_struct):
    """Turn a pymatgen Slab upside down, so its other face points along +z.

    `get_slabs` returns each asymmetric slab in one orientation, so without this the
    opposite termination cannot be built at all. Mirrors what fairchem's `compute_slabs`
    does with `is_structure_invertible`/`flip_struct`.
    """
    flipped = slab_struct.copy()
    # Mirror through z, then shift back into the cell. Fractional coords keep this exact.
    flipped = flipped.__class__(
        lattice=flipped.lattice,
        species=[site.species for site in flipped],
        coords=[[c[0], c[1], 1.0 - c[2]] for c in flipped.frac_coords],
        miller_index=slab_struct.miller_index,
        oriented_unit_cell=slab_struct.oriented_unit_cell,
        shift=slab_struct.shift,
        scale_factor=slab_struct.scale_factor,
        site_properties=flipped.site_properties,
    )
    return flipped


def _ensure_z_clearance(atoms, min_vacuum=MIN_VACUUM):
    """Set cell_z so the slab has `min_vacuum` of vacuum AND cell_z >= 2*span + 2.

    The second condition is not physics -- it is a defence against a bug in
    fairchem's adsorbate placement. `_get_scaled_normal`
    (fairchem/data/oc/core/adsorbate_slab_config.py) centres the adsorption site in
    the cell and calls wrap() to "not deal with pbc issues". Centring puts the slab
    in [cell_z/2 - span, cell_z/2], which only stays above z=0 when
    cell_z >= 2*span. Below that, the slab's own underside wraps around to sit
    ABOVE the site; the overlap solver then lifts H past the top of the cell, and
    `ocp_adslab_generator` folds it back down INTO the slab, where it neighbours a
    frozen atom and is discarded as `adsorbate_intercalated`.

    Measured against the pre-fix results the predictor `2*span - cell_z` tracked the
    observed rejection rate with r = 0.97 (Mo2N(001) +14.0 Å -> 100/100 rejected;
    Mo2C(100) +1.4 Å -> 21/100; Mo2C(110) -2.4 Å -> 0.2/100).

    With the OC20 convention the vacuum term dominates for every plain slab
    (span <= 15 Å, vacuum 20 Å), so this only ever binds for the thick ZSL
    interfaces. Applying it here, once, at the end of every builder, is what makes
    the invariant true by construction rather than by a downstream patch.
    """
    atoms = atoms.copy()
    span = _atom_span(atoms)
    atoms.cell[2, 2] = max(span + min_vacuum, 2.0 * span + 2.0)
    atoms.center(axis=2)
    if atoms.cell[2, 2] < 2.0 * span:
        raise ValueError(
            f"z clearance failed: cell_z {atoms.cell[2, 2]:.2f} Å < 2*span "
            f"{2 * span:.2f} Å -- adsorbate placement would wrap the slab onto itself"
        )
    return atoms


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


def create_mxene_basal_slab(min_ab=MIN_AB, vacuum=MIN_VACUUM):
    """One intact Ti3C2O2 sheet, tiled to `min_ab`.

    `create_ti3c2_bulk` returns exactly one O-Ti-C-Ti-C-Ti-O sheet (6 Å) plus its
    3 Å vdW gap, so tiling that cell in-plane and re-establishing the vacuum gives
    the basal slab directly. Routing it through create_slab would ask SlabGenerator
    to hit a 7 Å thickness target in a 9 Å cell, which either returns the same sheet
    or cuts through it.
    """
    sheet = _tile_to_min_ab(create_ti3c2_bulk(), min_ab)
    sheet.set_pbc([True, True, False])
    sheet = _ensure_z_clearance(sheet, min_vacuum=vacuum)
    _apply_constraints(sheet)
    return sheet


def create_graphene_sheet(size=None, vacuum=MIN_VACUUM, min_ab=MIN_AB):
    """Create a graphene sheet slab.

    Sized by the `min_ab` threshold, not by a fixed repeat count. The old hard-coded
    (4, 4, 1) cleared OC20's 8 Å only coincidentally -- 4 x 2.46 = 9.84 Å -- and would
    have violated it silently had `a` changed. `size` stays available as an explicit
    override for a deliberately larger cell. At a = 2.46 the derived tiling is exactly
    (4, 4, 1), so this changes no structure today; it is a guard, not a resize.
    """
    a = 2.46  # Å, graphene lattice constant (experiment and PBE agree; no change owed)

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

    sheet = bulk.repeat(size) if size is not None else _tile_to_min_ab(bulk, min_ab)
    sheet.set_pbc([True, True, False])
    return _ensure_z_clearance(sheet, min_vacuum=vacuum)


def create_n_doped_graphene(size=None, vacuum=MIN_VACUUM):
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


def create_slab(bulk_atoms, miller="(100)", min_ab=MIN_AB,
                min_thickness=MIN_SLAB_THICKNESS, vacuum=MIN_VACUUM, termination=0):
    """Create a surface slab for a requested Miller index, sized to the OC20 convention.

    Cuts to a target thickness in ÅNGSTRÖMS. The previous implementation passed a
    layer count to `ase.build.surface`, whose `layers` counts *oriented-unit-cell
    repeats*, not atomic planes -- so `layers=4` meant 16 atomic planes / 30 Å for
    Mo2N(001) but 9 planes / 11.8 Å for Mo2C(111). Nothing in the code said which,
    and the resulting 46 Å MoS2 "basal plane" was eight stacked monolayers.

    pymatgen's SlabGenerator is used with OC20's exact parameters, and is called on
    the bulk cell it was HANDED -- deliberately not routed through fairchem's
    `standardize_bulk`. SpacegroupAnalyzer standardization permutes Mo2C's axes
    (4.725, 6.022, 5.195 -> 4.725, 5.195, 6.022), which would silently redefine
    `Mo2C_(110)` as the plane we currently call (101). Verified equivalent to the old
    ase cut for that facet: ase (1,1,0) gives 7.65 x 5.20 in-plane, this gives
    5.20 x 7.65.

    `termination` indexes the distinct terminations available for the facet. Index 0 is
    the first `get_slabs` result -- arbitrary, but no more so than the single cut
    `ase.build.surface` used to return. Ranking them by relaxed energy is follow-up work.

    The list includes FLIPPED copies of any slab whose two faces differ, because
    `get_slabs` returns each asymmetric slab in one orientation only and the opposite
    termination is simply that slab upside down. Without the flip, one face of every
    asymmetric slab is unreachable -- e.g. Mo2C(100) and MoP(001) each expose exactly one
    `get_slabs` termination (anion-terminated), and their metal-terminated faces, which
    the old `ase.build.surface` cut happened to return, could not be built at all.
    fairchem's own `compute_slabs` does the same flip for the same reason.
    """
    indices = _parse_miller(miller)

    generator = SlabGenerator(
        AseAtomsAdaptor.get_structure(bulk_atoms),
        miller_index=indices,
        min_slab_size=min_thickness,
        min_vacuum_size=vacuum,
        lll_reduce=False,
        center_slab=True,
        primitive=True,
        max_normal_search=1,
    )
    # Transcribed verbatim from fairchem's OC20 call. NOTE: `max_broken_bonds=0` is a
    # no-op here and does NOT stop a covalent sandwich being sliced. pymatgen only
    # counts broken bonds when a `bonds` dict is supplied (pymatgen/core/surface.py:
    # `z_ranges = [] if bonds is None else get_z_ranges(bonds, ztol)`), so with
    # bonds=None the count stays 0 for every shift and EVERY termination is accepted.
    # What actually keeps the layered materials intact is that they bypass this
    # function entirely -- see create_tmd_basal_slab / create_mxene_basal_slab.
    # Passing a real `bonds` dict is the ambitious version; it would only matter if a
    # layered compound were ever routed through here.
    slabs = generator.get_slabs(tol=0.3, bonds=None, max_broken_bonds=0, symmetrize=False)
    if not slabs:
        raise ValueError(
            f"SlabGenerator found no {indices} termination at min_slab_size="
            f"{min_thickness} Å; facet may not be cleavable"
        )
    slabs = slabs + [_flip_slab_z(s) for s in slabs if not s.is_symmetric()]
    if termination >= len(slabs):
        raise ValueError(
            f"termination index {termination} out of range: {indices} has "
            f"{len(slabs)} termination(s) (including flipped faces of asymmetric slabs)"
        )

    slab = AseAtomsAdaptor.get_atoms(slabs[termination])
    slab = _tile_to_min_ab(slab, min_ab)
    slab.set_pbc([True, True, False])
    slab = _ensure_z_clearance(slab, min_vacuum=vacuum)

    thickness = _material_thickness(slab)
    if thickness < min_thickness - 0.5:
        raise ValueError(
            f"{indices} slab is only {thickness:.2f} Å of material "
            f"(< {min_thickness} Å); SlabGenerator under-cut the facet"
        )
    _apply_constraints(slab)
    return slab


def create_tmd_basal_slab(formula, a, thickness, min_ab=MIN_AB, vacuum=MIN_VACUUM):
    """A single 2H-MX2 monolayer in its HEXAGONAL cell, tiled to `min_ab`.

    The basal plane of a vdW crystal is one monolayer -- that is the model the TMD
    HER literature uses (3x3 or 4x4 of a monolayer, 12-15 Å vacuum). Cutting it with
    SlabGenerator instead would obey OC20's 7 Å minimum and hand back two
    monolayers, and the old `create_slab` route handed back EIGHT (46 Å of MoS2).

    Kept separate from `create_tmd_monolayer`, which returns a rectangular cell
    because edge ribbons need orthogonal axes to cut a clean finite-in-x strip. Here
    the hexagonal cell is the right one: it keeps the 3-fold symmetry of the basal
    plane, so the H site mesh is not biased by an artificial rectangular supercell.
    """
    sheet = mx2(formula=formula, kind='2H', a=a, thickness=thickness,
                size=(1, 1, 1), vacuum=7.5)
    sheet = _tile_to_min_ab(sheet, min_ab)
    sheet.set_pbc([True, True, False])
    sheet = _ensure_z_clearance(sheet, min_vacuum=vacuum)
    _apply_constraints(sheet)
    return sheet


def create_edge_ribbon(bulk_atoms, width=6, length=2, vacuum=MIN_VACUUM, edge_type="Mo",
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
    # Ribbons are excluded from the AdsorbML pipeline today (placement is z-oriented,
    # and a ribbon's active face points along x), but hold the same z invariant so they
    # are screenable the moment that changes.
    ribbon = _ensure_z_clearance(ribbon, min_vacuum=vacuum)
    _apply_constraints(ribbon)
    return ribbon


# Å below the slab's topmost atom that a defect site may sit and still count as
# surface. Measured separation on the current inputs is clean: every exposed dopant
# is within 0.8 Å of z_max, while the buried ones sit at 1.18 Å (Mo2C(100), under a
# C face) and 1.36 Å (Mo2N(112), under an N face).
BURIAL_TOL = 0.8


def _assert_surface_species(slab, symbol, target_idx, action, burial_tol=BURIAL_TOL):
    """Raise if the topmost `symbol` atom is buried beneath the slab's real surface.

    Each defect builder picks its site as the highest atom OF ITS OWN SPECIES, which is
    the right way to choose *which* atom to modify but says nothing about whether that
    atom is at the surface. On an anion-terminated facet the topmost metal sits a full
    layer down, so the defect is created under an intact anion sheet where no adsorbate
    can reach it.

    All three builders previously relied on `z_top = max(z[target_idx])` as their only
    guard, which cannot fail by construction -- the maximum is itself a member of the
    set it is compared against -- so their "refusing to dope a buried atom" branches
    were unreachable. This is the check that makes those docstrings true.
    """
    z = slab.get_positions()[:, 2]
    depth = float(z.max() - np.max(z[target_idx]))
    if depth > burial_tol:
        raise ValueError(
            f"topmost {symbol} atom sits {depth:.2f} A below the slab surface "
            f"(> {burial_tol} A), so {action} would bury the defect under another "
            f"species; this facet exposes a different termination -- pick a "
            f"termination whose surface contains {symbol}, or target the exposed species"
        )


def create_vacancy_slab(slab, vacancy_symbol, tol=0.5):
    """Remove one top-layer atom (of its own species' top layer) to create a vacancy."""
    positions = slab.get_positions()
    z_positions = positions[:, 2]

    target_idx = [i for i, atom in enumerate(slab) if atom.symbol == vacancy_symbol]
    if not target_idx:
        raise ValueError(f"No {vacancy_symbol} atoms in slab; cannot create a vacancy")

    _assert_surface_species(slab, vacancy_symbol, target_idx, "removing one")

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

    _assert_surface_species(slab, vacancy_symbol, target_idx, f"removing {count}")

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

    _assert_surface_species(slab, target_symbol, target_idx,
                            f"substituting {dopant_symbol}")

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
    # The cluster raises the atom span, so re-establish the z invariant here rather
    # than leaving each caller to remember it.
    return _ensure_z_clearance(slab)


def _build_zsl_interface(substrate_atoms, film_atoms, substrate_miller, film_miller,
                          separation=2.2, vacuum=MIN_VACUUM, strain_tol=0.02,
                          substrate_thickness=MIN_SLAB_THICKNESS,
                          film_thickness=MIN_SLAB_THICKNESS, in_layers=False,
                          max_atoms=MAX_ATOMS_INTERFACE):
    """Lattice-match a substrate/film pair with pymatgen ZSL and return a combined ASE slab.

    Replaces the old concatenate-then-set_cell(substrate.cell) approach, which never
    lattice-matched the two in-plane periodicities and wrapped mismatched atoms on
    top of each other (0.22-0.92 A overlaps).

    Thicknesses default to ANGSTROMS (`in_layers=False`), matching OC20's 7 A, rather
    than pymatgen's default of LAYERS. Passing layers is how the MoS2 interfaces ended
    up 31-39 A thick: `film_thickness=2` meant two 12.3 A bulk repeats, i.e. four
    monolayers. Callers that genuinely need a layer count -- the MXene, where one
    "layer" is one intact O-Ti-C-Ti-C-Ti-O sheet -- pass `in_layers=True` explicitly.

    `max_atoms` rejects coincidence cells above that size; the smallest-strain match
    that also fits is returned. It defaults to MAX_ATOMS_INTERFACE rather than OC22's
    250-atom ceiling, because a ZSL coincidence cell is set by the lattice mismatch and
    not by thickness -- at 250 every current Ni/MoX match would be rejected.
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
            in_layers=in_layers,
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
    too_narrow = 0
    for strain, interface, termination in candidates:
        atoms = AseAtomsAdaptor.get_atoms(interface)
        atoms.set_pbc([True, True, False])
        if max_atoms is not None and len(atoms) > max_atoms:
            too_big += 1
            continue
        # ZSL happily returns extremely elongated coincidence cells -- Ni/MoB(111)'s
        # lowest-strain match is 4.31 x 51.07 Å, which would put periodic H images
        # 4.3 Å apart. Skip to the next-lowest-strain match rather than accept a cell
        # too narrow to hold an isolated adsorbate.
        if min(np.linalg.norm(atoms.cell[0]), np.linalg.norm(atoms.cell[1])) < MIN_AB - 0.01:
            too_narrow += 1
            continue
        if _min_covalent_radius_ratio(atoms) >= 1.0:
            # An interface is substrate + film stacked, so it is the one family thick
            # enough that `vacuum_over_film` alone can leave 2*span > cell_z. This is
            # where _ensure_z_clearance's second branch actually does work.
            atoms = _ensure_z_clearance(atoms, min_vacuum=vacuum)
            print(f"[ZSL termination={termination} strain={strain:.3%} natoms={len(atoms)}] ", end="")
            return atoms

    if not candidates:
        raise ValueError(
            f"No ZSL lattice match within {strain_tol:.1%} strain for "
            f"substrate_miller={substrate_miller}, film_miller={film_miller} "
            f"(mismatched lattices; refusing to emit an overlapping structure)"
        )
    if too_big + too_narrow == len(candidates):
        raise ValueError(
            f"All {len(candidates)} ZSL match(es) within {strain_tol:.1%} strain for "
            f"substrate_miller={substrate_miller}, film_miller={film_miller} were "
            f"rejected: {too_big} exceed the {max_atoms}-atom cap, {too_narrow} are "
            f"narrower than min_ab {MIN_AB} Å in one direction"
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
    ni_bulk = ase_bulk("Ni", "fcc", a=NI_A, cubic=True)
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

    This is the one caller that passes thicknesses in LAYERS rather than Å: one
    "layer" of the film is one intact O-Ti-C-Ti-C-Ti-O sheet, which is exactly the
    quantity we want to hold at 1, and an Å target would cut through it.

    The substrate thickness is therefore ALSO in layers, and must be read per facet
    from `_NI_SUBSTRATE_LAYERS` -- a layer is an oriented-cell repeat, not an atomic
    plane, so a single number means different amounts of material on (100) and (111).
    A flat `substrate_thickness=4` used to mean 8 planes / 14.08 Å on (100), double the
    intended ~7 Å and 60 % of that cell's atoms, while meaning a correct 4 planes /
    8.13 Å on (111). This is the same layers-vs-planes trap `_build_zsl_interface`
    warns about for the film, biting the substrate in the same call.
    """
    ni_bulk = ase_bulk("Ni", "fcc", a=NI_A, cubic=True)
    mxene_bulk = create_ti3c2_bulk()
    indices = _parse_miller(miller)
    substrate_layers = _NI_SUBSTRATE_LAYERS.get(indices)
    if substrate_layers is None:
        raise ValueError(
            f"Ni substrate layer count not tabulated for facet {indices}; add it to "
            f"_NI_SUBSTRATE_LAYERS (smallest layer count clearing "
            f"{MIN_SLAB_THICKNESS} Å) rather than guessing a shared value"
        )
    interface = _build_zsl_interface(
        ni_bulk, mxene_bulk, substrate_miller=indices, film_miller=(0, 0, 1),
        separation=separation, strain_tol=strain_tol,
        in_layers=True, film_thickness=1, substrate_thickness=substrate_layers,
        max_atoms=max_atoms,
    )
    _assert_substrate_facet(interface, "Ni", indices, label=f"Ni/MXene {miller}")
    # One intact O-Ti-C-Ti-C-Ti-O sheet: no >3 Å gap within the film sublattice.
    _assert_film_integrity(interface, substrate_symbol="Ni", max_internal_gap=3.0,
                           label=f"Ni/MXene {miller}")
    _freeze_substrate_bottom_half(interface, substrate_symbol="Ni")
    return interface


def create_ni_on_graphene(ni_atoms=4, height=1.8, size=None, vacuum=MIN_VACUUM):
    """Create Ni cluster on graphene sheet."""
    sheet = create_graphene_sheet(size=size, vacuum=vacuum)
    sheet = add_cluster_on_surface(sheet, "Ni", n_atoms=ni_atoms, height=height)
    _apply_constraints(sheet)
    return sheet


def create_ni_on_n_doped_graphene(ni_atoms=4, height=1.8, size=None, vacuum=MIN_VACUUM):
    """Create Ni cluster on N-doped graphene sheet."""
    sheet = create_n_doped_graphene(size=size, vacuum=vacuum)
    sheet = add_cluster_on_surface(sheet, "Ni", n_atoms=ni_atoms, height=height)
    _apply_constraints(sheet)
    return sheet


def create_graphene_nanoribbon(width=6, length=3, vacuum=MIN_VACUUM):
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
    ribbon.set_pbc([False, True, False])
    ribbon = _ensure_z_clearance(ribbon, min_vacuum=vacuum)
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
# this pass; pymatgen's SlabGenerator cuts these Miller strings correctly regardless
# of crystal system, so no parser change is needed for any of them).
#
# Miller indices are interpreted in the basis of the bulk cell each create_*_bulk()
# returns -- create_slab deliberately does not standardize the cell first, because
# SpacegroupAnalyzer standardization swaps Mo2C's b and c axes and would silently
# redefine (110) as the plane we call (101).
DEFAULT_MILLERS = ['(100)', '(110)', '(111)']
FACETS = {
    'MoS2': ['(001)'],
    'MoSe2': ['(001)'],
    'MoP': ['(001)'],
    'Mo2N': ['(001)', '(100)', '(111)', '(112)'],
}

# In-plane width for defect/dopant slabs, in Å of defect-image separation.
#
# Deliberately the same as MIN_AB. A larger threshold buys nothing here because
# tiling is discrete: MoB(110)'s primitive surface vector is 8.76 Å, so demanding
# 12 Å jumps it to 17.5 Å and 192 atoms. At 8 Å every defect cell lands between
# 8.0 and 14.0 Å of separation for 27-144 atoms, which is the range the
# single-dopant/vacancy HER literature uses (the MoS2 S-vacancy work uses 3x3,
# ~9.5 Å), and no structure exceeds MAX_ATOMS_TARGET.
#
# The trade-off is coverage: Mo2N(001) at 2x2 is one dopant per four top-layer Mo,
# which reads as a doped surface more than an isolated dopant. Raise this to 10.0
# if a dopant-image convergence check shows it matters. Note this does NOT fix the
# separate finding that the best H site lands far from the dopant -- that needs
# site generation restricted to a radius around the defect, in AdsorbML step 2.
MIN_AB_DEFECT = MIN_AB

# 2H-TMD monolayer params: (in-plane a, S-S vertical thickness) Å.
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

        # The 2H-TMD basal plane is ONE monolayer, not a thickness-cut slab: it is a
        # vdW crystal, so any "at least 7 Å" rule hands back a stack of sheets held
        # together by nothing. (The old layer-count route handed back eight.)
        if formula in TMD_PARAMS:
            tmd_a, tmd_t = TMD_PARAMS[formula]
            for miller in FACETS[formula]:
                write_structure(f"{formula}_{miller}",
                    lambda f=formula, a=tmd_a, t=tmd_t: create_tmd_basal_slab(f, a=a, thickness=t))
            continue

        for miller in FACETS.get(formula, DEFAULT_MILLERS):
            write_structure(f"{formula}_{miller}",
                lambda m=miller, b=bulk: create_slab(b.copy(), miller=m))

    # ── Part 2: Chalcogenide variants (vacancies, edges) ──────────
    for formula, vac_sym in chalcogenides.items():
        print(f"\n{formula} variants:")
        facets = FACETS[formula]
        tmd_a, tmd_t = TMD_PARAMS[formula]
        print(f"    [defect cell] min_ab {MIN_AB_DEFECT} A of defect-image separation")

        for miller in facets:
            # Vacancies are cut into the same single monolayer as the pristine basal
            # slab, so pristine and defected differ only by the missing atom.
            write_structure(f"{formula}_{miller}_vac{vac_sym}",
                lambda f=formula, a=tmd_a, t=tmd_t: create_vacancy_slab(
                    create_tmd_basal_slab(f, a=a, thickness=t, min_ab=MIN_AB_DEFECT), vac_sym))
            write_structure(f"{formula}_{miller}_vac2{vac_sym}",
                lambda f=formula, a=tmd_a, t=tmd_t: create_multi_vacancy_slab(
                    create_tmd_basal_slab(f, a=a, thickness=t, min_ab=MIN_AB_DEFECT),
                    vac_sym, count=2))

        # Edge ribbons cut from a single MONOLAYER (layers_z=1), not the 2-sheet
        # bulk -- the old ribbons were accidental bilayer rods. Mo-edge and
        # X(=S/Se)-edge are the two zigzag terminations; the Mo-edge is the
        # literature HER-active site.
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
        print(f"    [defect cell] min_ab {MIN_AB_DEFECT} A of defect-image separation")

        for miller in facets:
            # Anion vacancy
            write_structure(f"{formula}_{miller}_vac{anion}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, min_ab=MIN_AB_DEFECT), anion))
            # Metal vacancy
            write_structure(f"{formula}_{miller}_vac{metal}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, min_ab=MIN_AB_DEFECT), metal))
            # Double vacancies
            write_structure(f"{formula}_{miller}_vac2{anion}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, min_ab=MIN_AB_DEFECT), anion, count=2))
            write_structure(f"{formula}_{miller}_vac2{metal}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m, min_ab=MIN_AB_DEFECT), metal, count=2))
            # Metal-site dopants
            for dopant in dopants:
                write_structure(f"{formula}_{miller}_dop{dopant}",
                    lambda m=miller, d=dopant: create_substitution_slab(
                        create_slab(bulk.copy(), miller=m, min_ab=MIN_AB_DEFECT), metal, d))

        # Edge ribbons for Mo2C and MoB
        if formula in ('Mo2C', 'MoB'):
            for edge_type, label in [("Mo", "Mo"), ("X", anion)]:
                write_structure(f"{formula}_edge_{label}",
                    lambda et=edge_type: create_edge_ribbon(bulk.copy(), edge_type=et))
                write_structure(f"{formula}_edge_{label}_large",
                    lambda et=edge_type: create_edge_ribbon(bulk.copy(), width=10, length=3, edge_type=et))

        # The `_sheet` family (a 4x4 slab at 4x the atoms) is not emitted any more.
        # It existed as a numerical yardstick and it did its job: it reproduced the
        # 2x2 value to 1-3 meV, confirming the smaller in-plane cell is converged.
        # Keeping 8 such structures in a campaign that is GPU-bound buys nothing.
        # (graphene_sheet is unrelated despite the name -- it is a real structure.)

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
    # vacuum), so only (001) with a single sheet is emitted.
    #
    # Built by tiling the bulk cell directly rather than through create_slab: the
    # bulk cell already IS one O-Ti-C-Ti-C-Ti-O sheet plus its vdW gap, so a
    # thickness-targeted cut would either return the same thing or slice the sheet.
    write_structure("Ti3C2O2_(001)",
        lambda: create_mxene_basal_slab())

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


# Families that are legitimately thinner than MIN_SLAB_THICKNESS or narrower than
# MIN_AB in one direction, and why. Anything not listed here must meet both.
#   *_edge_*, *nanoribbon* : finite in x on purpose -- x is vacuum, not a cell width
#   MoS2/MoSe2/Ti3C2/graphene basal : single sheets; a thickness floor would stack them
_THIN_OK = ("_edge_", "nanoribbon", "graphene", "MoS2_(001)", "MoSe2_(001)", "Ti3C2O2_(001)")
_NARROW_OK = ("_edge_", "nanoribbon", "on_nanoribbon")


def _assert_sizing_invariants(name, slab):
    """Gate every emitted structure on the OC20 sizing convention.

    The placement-wrap check is the one that matters: it is the invariant whose
    violation silently produced zero H* candidates for all 48 Mo2N structures, and
    it was invisible because nothing downstream looked at cell geometry. Checking it
    at write time means a regression in any builder fails the generator, not a GPU run.
    """
    span = _atom_span(slab)
    cell_z = float(slab.cell[2, 2])
    if cell_z < 2.0 * span + 1.0:
        raise ValueError(
            f"placement-wrap invariant violated: cell_z {cell_z:.2f} Å < 2*span + 1 "
            f"({2 * span + 1.0:.2f} Å). fairchem would fold the adsorbate into the slab; "
            f"see _ensure_z_clearance"
        )

    la, lb = (float(np.linalg.norm(slab.cell[i])) for i in (0, 1))
    if min(la, lb) < MIN_AB - 0.01 and not any(k in name for k in _NARROW_OK):
        raise ValueError(
            f"in-plane width {la:.2f} x {lb:.2f} Å is below OC20's min_ab {MIN_AB} Å"
        )

    thickness = _material_thickness(slab)
    if thickness < MIN_SLAB_THICKNESS - 0.5 and not any(k in name for k in _THIN_OK):
        raise ValueError(
            f"material thickness {thickness:.2f} Å is below OC20's "
            f"min_slab_size {MIN_SLAB_THICKNESS} Å"
        )

    cap = MAX_ATOMS_INTERFACE if "interface" in name else MAX_ATOMS_TARGET
    if len(slab) > cap:
        print(f"[{len(slab)} atoms > {cap} target] ", end="")


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
        _assert_sizing_invariants(name, slab)
        if dry_run:
            print(f"✓ (dry-run, {len(slab)} atoms, not written)")
            return
        dir_name.mkdir(parents=True, exist_ok=True)
        write(str(poscar_file), slab, format='vasp')
        print(f"✓ ({len(slab)} atoms)")
    except Exception as e:
        print(f"✗ Error: {e}")
        # Only ever remove a POSCAR this call was actually trying to write. The unlink
        # used to be unconditional, so `--dry-run` -- whose entire contract is to touch
        # nothing -- deleted the existing input for every structure that failed to build.
        # That is exactly the "never clobber data/" rule the repo warns about.
        if not dry_run and poscar_file.exists():
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
