"""
Generate POSCAR files for Mo compounds
Creates surface slabs using literature lattice parameters and ASE
"""

import os
from pathlib import Path
from ase import Atoms
from ase.io import write
from ase.constraints import FixAtoms
from ase.build import fcc111, fcc100, surface
import numpy as np


def create_mos2_bulk():
    """Create MoS2 bulk structure (hexagonal)"""
    # Experimental lattice parameters
    a = 3.160  # Å
    c = 12.295  # Å
    
    # Hexagonal cell
    cell = np.array([
        [a, 0, 0],
        [-a/2, a*np.sqrt(3)/2, 0],
        [0, 0, c]
    ])
    
    # Mo at (0, 0, 1/2), S at (1/3, 2/3, z) and (2/3, 1/3, -z)
    symbols = ['Mo', 'S', 'S']
    positions = np.array([
        [0.0, 0.0, 0.5],
        [1/3, 2/3, 0.62],
        [2/3, 1/3, 0.38],
    ])
    
    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_mose2_bulk():
    """Create MoSe2 bulk structure (hexagonal)"""
    a = 3.289  # Å
    c = 12.995  # Å
    
    cell = np.array([
        [a, 0, 0],
        [-a/2, a*np.sqrt(3)/2, 0],
        [0, 0, c]
    ])
    
    symbols = ['Mo', 'Se', 'Se']
    positions = np.array([
        [0.0, 0.0, 0.5],
        [1/3, 2/3, 0.62],
        [2/3, 1/3, 0.38],
    ])
    
    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_mop_bulk():
    """Create MoP bulk structure (cubic)"""
    a = 3.240  # Å
    
    cell = np.eye(3) * a
    
    symbols = ['Mo', 'P']
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    
    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_mo2n_bulk():
    """Create Mo2N bulk structure (cubic anti-perovskite)"""
    a = 4.160  # Å
    
    cell = np.eye(3) * a
    
    symbols = ['Mo', 'Mo', 'N']
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
    ])
    
    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_mo2c_bulk():
    """Create Mo2C bulk structure (orthorhombic, Pbcn)."""
    a, b, c = 4.732, 6.037, 5.204  # Å, experimental

    cell = np.array([
        [a, 0, 0],
        [0, b, 0],
        [0, 0, c],
    ])

    # Pbcn Mo2C: 4 Mo + 2 C per unit cell (Wyckoff 8d Mo, 4c C)
    symbols = ['Mo', 'Mo', 'Mo', 'Mo', 'C', 'C']
    positions = np.array([
        [0.25, 0.12, 0.08],
        [0.75, 0.88, 0.92],
        [0.25, 0.62, 0.42],
        [0.75, 0.38, 0.58],
        [0.0,  0.36, 0.25],
        [0.5,  0.64, 0.75],
    ])

    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_mob_bulk():
    """Create MoB bulk structure (tetragonal, I4_1/amd)."""
    a = 3.110  # Å
    c = 16.950  # Å

    cell = np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, c],
    ])

    # I4_1/amd MoB: 8 Mo + 8 B per conventional cell
    symbols = ['Mo'] * 8 + ['B'] * 8
    positions = np.array([
        # Mo 8e positions
        [0.0,  0.0,  0.197],
        [0.5,  0.5,  0.447],
        [0.0,  0.5,  0.697],
        [0.5,  0.0,  0.947],
        [0.0,  0.0,  0.803],
        [0.5,  0.5,  0.553],
        [0.0,  0.5,  0.303],
        [0.5,  0.0,  0.053],
        # B 8e positions
        [0.0,  0.0,  0.348],
        [0.5,  0.5,  0.598],
        [0.0,  0.5,  0.848],
        [0.5,  0.0,  0.098],
        [0.0,  0.0,  0.652],
        [0.5,  0.5,  0.402],
        [0.0,  0.5,  0.152],
        [0.5,  0.0,  0.902],
    ])

    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_ti3c2_bulk():
    """Create Ti3C2 MXene bulk structure (hexagonal, O-terminated: Ti3C2O2)."""
    a = 3.071  # Å
    c = 20.0   # Å (large c for layered structure with vacuum)

    cell = np.array([
        [a, 0, 0],
        [-a / 2, a * np.sqrt(3) / 2, 0],
        [0, 0, c],
    ])

    # Ti3C2O2 monolayer centred in cell
    symbols = ['O', 'Ti', 'C', 'Ti', 'C', 'Ti', 'O']
    # z positions in fractional coords, symmetric about 0.5
    positions = np.array([
        [1 / 3, 2 / 3, 0.350],  # O bottom
        [2 / 3, 1 / 3, 0.400],  # Ti bottom
        [0.0,   0.0,   0.450],  # C bottom
        [1 / 3, 2 / 3, 0.500],  # Ti middle
        [2 / 3, 1 / 3, 0.550],  # C top
        [0.0,   0.0,   0.600],  # Ti top
        [1 / 3, 2 / 3, 0.650],  # O top
    ])

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


def create_edge_ribbon(bulk_atoms, width=6, length=2, vacuum=8, edge_type="Mo"):
    """Create a simple edge ribbon by adding vacuum in x and z.

    edge_type:
        "Mo"  -> remove chalcogen atoms at ribbon edges
        "X"   -> remove Mo atoms at ribbon edges (X = S or Se)
    """
    ribbon = bulk_atoms.repeat((width, length, 1))

    # Create vacuum in x and z to form edges and a single-layer ribbon
    ribbon.center(vacuum=vacuum, axis=0)
    ribbon.center(vacuum=vacuum, axis=2)
    ribbon.set_pbc([False, True, False])

    # Determine edge atoms by x position
    positions = ribbon.get_positions()
    x_positions = positions[:, 0]
    x_min = np.min(x_positions)
    x_max = np.max(x_positions)
    tol = 0.3  # Angstroms
    edge_mask = (x_positions - x_min < tol) | (x_max - x_positions < tol)

    # Remove atoms at edges to approximate termination
    if edge_type == "Mo":
        remove_symbols = {"S", "Se"}
    else:
        remove_symbols = {"Mo"}

    remove_indices = [
        i for i, atom in enumerate(ribbon)
        if edge_mask[i] and atom.symbol in remove_symbols
    ]
    if remove_indices:
        del ribbon[remove_indices]

    _apply_constraints(ribbon)
    return ribbon


def create_vacancy_slab(slab, vacancy_symbol):
    """Remove one top-layer atom to create a vacancy."""
    positions = slab.get_positions()
    z_positions = positions[:, 2]
    z_max = np.max(z_positions)
    tol = 0.5  # Angstroms

    candidates = [
        i for i, atom in enumerate(slab)
        if atom.symbol == vacancy_symbol and (z_max - z_positions[i]) < tol
    ]
    if not candidates:
        return slab

    # Remove the candidate closest to xy center
    center_xy = np.mean(positions[:, :2], axis=0)
    distances = [np.linalg.norm(positions[i, :2] - center_xy) for i in candidates]
    remove_index = candidates[int(np.argmin(distances))]
    del slab[remove_index]
    return slab


def create_multi_vacancy_slab(slab, vacancy_symbol, count=2):
    """Remove multiple top-layer atoms to create vacancy clusters."""
    positions = slab.get_positions()
    z_positions = positions[:, 2]
    z_max = np.max(z_positions)
    tol = 0.5  # Angstroms

    candidates = [
        i for i, atom in enumerate(slab)
        if atom.symbol == vacancy_symbol and (z_max - z_positions[i]) < tol
    ]
    if len(candidates) < count:
        return slab

    center_xy = np.mean(positions[:, :2], axis=0)
    distances = [np.linalg.norm(positions[i, :2] - center_xy) for i in candidates]
    sorted_candidates = [c for _, c in sorted(zip(distances, candidates))]

    remove_indices = sorted_candidates[:count]
    for idx in sorted(remove_indices, reverse=True):
        del slab[idx]
    return slab


def create_substitution_slab(slab, target_symbol, dopant_symbol):
    """Substitute one top-layer atom with a dopant."""
    positions = slab.get_positions()
    z_positions = positions[:, 2]
    z_max = np.max(z_positions)
    tol = 0.5  # Angstroms

    candidates = [
        i for i, atom in enumerate(slab)
        if atom.symbol == target_symbol and (z_max - z_positions[i]) < tol
    ]
    if not candidates:
        return slab

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


def create_ni_slab(miller="(111)", size=(3, 3, 4), vacuum=8):
    """Create a simple Ni slab for interface models."""
    a = 3.52  # Angstrom, fcc Ni
    if miller == "(100)":
        slab = fcc100("Ni", size=size, a=a, vacuum=vacuum)
    else:
        slab = fcc111("Ni", size=size, a=a, vacuum=vacuum)
    slab.set_pbc([True, True, False])
    return slab


def create_ni_mox_interface(mox_bulk_builder, miller="(111)", separation=2.2):
    """Create a simple Ni/MoX interface slab (trend-level model).

    Args:
        mox_bulk_builder: callable returning bulk Atoms (e.g. create_mo2n_bulk)
        miller: Miller index string
        separation: gap between Ni top and MoX bottom (Å)
    """
    ni = create_ni_slab(miller=miller, size=(3, 3, 4), vacuum=8)
    mox_bulk = mox_bulk_builder()
    mox = create_slab(mox_bulk, miller=miller, size=(2, 2, 2), vacuum=0)
    mox.set_pbc([True, True, False])

    # Stack MoX on top of Ni
    ni_positions = ni.get_positions()
    ni_top = np.max(ni_positions[:, 2])

    mox_positions = mox.get_positions()
    mox_shift = ni_top + separation - np.min(mox_positions[:, 2])
    mox.translate([0.0, 0.0, mox_shift])

    # Define combined cell with additional vacuum
    cell = ni.cell.copy()
    cell[2, 2] = np.max(mox.get_positions()[:, 2]) + 8.0

    interface = ni + mox
    interface.set_cell(cell)
    interface.set_pbc([True, True, False])

    # Freeze bottom half of Ni atoms only
    ni_indices = [i for i, atom in enumerate(interface) if atom.symbol == "Ni"]
    ni_z = interface.get_positions()[ni_indices, 2]
    ni_z_mid = (np.min(ni_z) + np.max(ni_z)) / 2
    fixed_indices = [i for i in ni_indices if interface[i].z < ni_z_mid]
    interface.set_constraint(FixAtoms(indices=fixed_indices))
    return interface


# Keep the old name as an alias for backward compatibility
def create_ni_mo2n_interface(miller="(111)", separation=2.2):
    """Create Ni/Mo2N interface (backward-compatible wrapper)."""
    return create_ni_mox_interface(create_mo2n_bulk, miller=miller, separation=separation)


def create_ni_mxene_interface(miller="(111)", separation=2.2):
    """Create Ni on Ti3C2O2 MXene interface."""
    ni = create_ni_slab(miller=miller, size=(3, 3, 4), vacuum=8)
    mxene = create_ti3c2_bulk()
    # Use a 3x3 supercell of MXene for size-matching with Ni slab
    mxene = mxene.repeat((3, 3, 1))
    mxene.set_pbc([True, True, False])

    ni_positions = ni.get_positions()
    ni_top = np.max(ni_positions[:, 2])

    mxene_positions = mxene.get_positions()
    mxene_shift = ni_top + separation - np.min(mxene_positions[:, 2])
    mxene.translate([0.0, 0.0, mxene_shift])

    cell = ni.cell.copy()
    cell[2, 2] = np.max(mxene.get_positions()[:, 2]) + 8.0

    interface = ni + mxene
    interface.set_cell(cell)
    interface.set_pbc([True, True, False])

    ni_indices = [i for i, atom in enumerate(interface) if atom.symbol == "Ni"]
    ni_z = interface.get_positions()[ni_indices, 2]
    ni_z_mid = (np.min(ni_z) + np.max(ni_z)) / 2
    fixed_indices = [i for i in ni_indices if interface[i].z < ni_z_mid]
    interface.set_constraint(FixAtoms(indices=fixed_indices))
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


def create_interface_with_dopant_generic(mox_bulk_builder, miller, dopant, target_symbol="Ni"):
    """Create Ni/MoX interface with a single dopant on the Ni top layer."""
    interface = create_ni_mox_interface(mox_bulk_builder, miller=miller)
    interface = create_substitution_slab(interface, target_symbol, dopant)
    return interface


def create_interface_with_cluster_generic(mox_bulk_builder, miller, dopant, cluster_size):
    """Create Ni/MoX interface with a small dopant cluster on surface."""
    interface = create_ni_mox_interface(mox_bulk_builder, miller=miller)
    interface = add_cluster_on_surface(interface, dopant, n_atoms=cluster_size)
    return interface


def create_interface_with_dopant(miller, dopant, target_symbol="Ni"):
    """Create Ni/Mo2N interface with a single dopant on the Ni top layer."""
    interface = create_ni_mo2n_interface(miller=miller)
    interface = create_substitution_slab(interface, target_symbol, dopant)
    return interface


def create_interface_with_cluster(miller, dopant, cluster_size):
    """Create Ni/Mo2N interface with a small dopant cluster on Ni surface."""
    interface = create_ni_mo2n_interface(miller=miller)
    interface = add_cluster_on_surface(interface, dopant, n_atoms=cluster_size)
    return interface


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUTS = REPO_ROOT / "data" / "inputs" / "VASP_inputs"


def generate_all_structures():
    """Generate all structure files"""
    
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

    # Systems that get Ni/MoX interface treatment
    interface_systems = {
        'Ni_Mo2N': create_mo2n_bulk,
        'Ni_Mo2C': create_mo2c_bulk,
        'Ni_MoB':  create_mob_bulk,
        'Ni_MoS2': create_mos2_bulk,
    }

    dopants = ["Pt", "Pd", "Ir", "Ru", "Ag", "Au", "Ni"]
    # Anna's decoration list (subset for interfaces)
    decorations = ["Ag", "Au", "Pd", "Pt", "Ir"]
    
    millers = ['(100)', '(110)', '(111)']
    interface_millers = ['(111)', '(100)']
    
    print("\n" + "="*60)
    print("Generating Structure Files for GPAW Calculations")
    print("="*60)
    
    base_dir = DATA_INPUTS
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Part 1: Basic slabs for all compounds ────────────────────
    for formula, builder in builders.items():
        print(f"\n{formula}:")
        
        try:
            bulk = builder()
            print(f"  Bulk structure: {len(bulk)} atoms/cell")
            
            for miller in millers:
                dir_name = base_dir / f"{formula}_{miller}"
                dir_name.mkdir(parents=True, exist_ok=True)
                poscar_file = dir_name / "POSCAR"
                print(f"    {miller}: ", end="", flush=True)
                try:
                    slab = create_slab(bulk.copy(), miller=miller)
                    write(str(poscar_file), slab, format='vasp')
                    print(f"✓ ({len(slab)} atoms)")
                except Exception as e:
                    print(f"✗ Error: {e}")
        except Exception as e:
            print(f"  ✗ Failed to create {formula}: {e}")

    # ── Part 2: Chalcogenide variants (vacancies, edges, sheets) ─
    for formula, vac_sym in chalcogenides.items():
        print(f"\n{formula} variants:")
        bulk = builders[formula]()

        for miller in millers:
            # Single vacancy
            _write_structure(base_dir, f"{formula}_{miller}_vac{vac_sym}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), vac_sym))
            # Double vacancy
            _write_structure(base_dir, f"{formula}_{miller}_vac2{vac_sym}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), vac_sym, count=2))
            # Nanosheet
            _write_structure(base_dir, f"{formula}_{miller}_sheet",
                lambda m=miller: create_slab(bulk.copy(), miller=m, size=(4, 4, 4), vacuum=10))

        # Edge ribbons
        for edge_type, label in [("Mo", "Mo"), ("X", vac_sym)]:
            _write_structure(base_dir, f"{formula}_edge_{label}",
                lambda et=edge_type: create_edge_ribbon(bulk.copy(), edge_type=et))
            _write_structure(base_dir, f"{formula}_edge_{label}_large",
                lambda et=edge_type: create_edge_ribbon(bulk.copy(), width=10, length=3, edge_type=et))

    # ── Part 3: Dopant/vacancy compounds (Mo2N, Mo2C, MoB) ──────
    for formula, info in dopant_compounds.items():
        print(f"\n{formula} dopants/vacancies:")
        bulk = builders[formula]()
        metal, anion = info['metal'], info['anion']

        for miller in millers:
            # Anion vacancy
            _write_structure(base_dir, f"{formula}_{miller}_vac{anion}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), anion))
            # Metal vacancy
            _write_structure(base_dir, f"{formula}_{miller}_vac{metal}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), metal))
            # Double vacancies
            _write_structure(base_dir, f"{formula}_{miller}_vac2{anion}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), anion, count=2))
            _write_structure(base_dir, f"{formula}_{miller}_vac2{metal}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), metal, count=2))
            # Metal-site dopants
            for dopant in dopants:
                _write_structure(base_dir, f"{formula}_{miller}_dop{dopant}",
                    lambda m=miller, d=dopant: create_substitution_slab(
                        create_slab(bulk.copy(), miller=m), metal, d))

        # Edge ribbons for Mo2C and MoB
        if formula in ('Mo2C', 'MoB'):
            for edge_type, label in [("Mo", "Mo"), ("X", anion)]:
                _write_structure(base_dir, f"{formula}_edge_{label}",
                    lambda et=edge_type: create_edge_ribbon(bulk.copy(), edge_type=et))
                _write_structure(base_dir, f"{formula}_edge_{label}_large",
                    lambda et=edge_type: create_edge_ribbon(bulk.copy(), width=10, length=3, edge_type=et))

            # Nanosheet
            for miller in millers:
                _write_structure(base_dir, f"{formula}_{miller}_sheet",
                    lambda m=miller: create_slab(bulk.copy(), miller=m, size=(4, 4, 4), vacuum=10))

    # ── Part 4: Ni/MoX interfaces ────────────────────────────────
    for sys_name, mox_builder in interface_systems.items():
        print(f"\n{sys_name} interfaces:")

        for miller in interface_millers:
            # Pristine interface
            _write_structure(base_dir, f"{sys_name}_interface_{miller}",
                lambda m=miller, b=mox_builder: create_ni_mox_interface(b, miller=m))

            # Single-atom dopants on Ni layer
            for dopant in dopants:
                _write_structure(base_dir, f"{sys_name}_interface_{miller}_dop{dopant}",
                    lambda m=miller, b=mox_builder, d=dopant:
                        create_interface_with_dopant_generic(b, m, d))

            # Noble metal clusters (2 and 4 atoms)
            for dec in decorations:
                for cs in [2, 4]:
                    _write_structure(base_dir, f"{sys_name}_interface_{miller}_cluster{cs}{dec}",
                        lambda m=miller, b=mox_builder, d=dec, s=cs:
                            create_interface_with_cluster_generic(b, m, d, s))

    # ── Part 5: MXene Ti3C2 + Ni ─────────────────────────────────
    print("\nNi/MXene Ti3C2:")

    # Bare MXene slabs
    mxene_bulk = create_ti3c2_bulk()
    for miller in millers:
        _write_structure(base_dir, f"Ti3C2O2_{miller}",
            lambda m=miller: create_slab(mxene_bulk.copy(), miller=m))

    # Ni/MXene interfaces
    for miller in interface_millers:
        _write_structure(base_dir, f"Ni_Ti3C2O2_interface_{miller}",
            lambda m=miller: create_ni_mxene_interface(miller=m))
        # Decorations on Ni/MXene
        for dec in decorations:
            for cs in [2, 4]:
                _write_structure(base_dir, f"Ni_Ti3C2O2_interface_{miller}_cluster{cs}{dec}",
                    lambda m=miller, d=dec, s=cs: add_cluster_on_surface(
                        create_ni_mxene_interface(miller=m), d, n_atoms=s))

    # ── Part 6: Ni on carbon variants ────────────────────────────
    print("\nNi on carbon:")

    # Pristine graphene
    _write_structure(base_dir, "graphene_sheet",
        lambda: create_graphene_sheet())

    # N-doped graphene
    _write_structure(base_dir, "graphene_N_doped",
        lambda: create_n_doped_graphene())

    # Graphene nanoribbon (CNT approximation)
    _write_structure(base_dir, "graphene_nanoribbon",
        lambda: create_graphene_nanoribbon())

    # Ni on graphene
    for n_ni in [2, 4]:
        _write_structure(base_dir, f"Ni{n_ni}_on_graphene",
            lambda n=n_ni: create_ni_on_graphene(ni_atoms=n))
        _write_structure(base_dir, f"Ni{n_ni}_on_graphene_N_doped",
            lambda n=n_ni: create_ni_on_n_doped_graphene(ni_atoms=n))

    # Ni on nanoribbon
    _write_structure(base_dir, "Ni4_on_nanoribbon",
        lambda: add_cluster_on_surface(
            create_graphene_nanoribbon(), "Ni", n_atoms=4))

    print("\n" + "="*60)
    print("✓ Structure generation complete!")
    print("="*60)


def _write_structure(base_dir, name, builder_fn):
    """Helper: build a structure and write its POSCAR."""
    dir_name = base_dir / name
    dir_name.mkdir(parents=True, exist_ok=True)
    poscar_file = dir_name / "POSCAR"
    print(f"    {name}: ", end="", flush=True)
    try:
        slab = builder_fn()
        write(str(poscar_file), slab, format='vasp')
        print(f"✓ ({len(slab)} atoms)")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == '__main__':
    generate_all_structures()
    print("\n✓ Done! All POSCAR files created.")
