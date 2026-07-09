"""
GPAW-based H Adsorption Energy Calculator
Calculate ΔGH for Mo compounds using quantum physics simulation

Requirements:
- GPAW (pip install gpaw) ✓ Already installed
- ASE (comes with GPAW)

Features:
- BFGS relaxation (fmax 0.03 eV/Å, up to 200 steps) matching the OC20/RPBE reference
- Plane-wave PW(350 eV) / RPBE with the OC20 Monkhorst-Pack k-mesh
- Auto-detecting parallelism (scales to available CPU/RAM)
- Incremental CSV writing (crash-safe)
- Checkpoint/resume (skips already-completed entries)
- Graceful shutdown on SIGINT/SIGTERM
"""

import os
import csv
import json
import signal
import fcntl
import argparse
import fnmatch
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from ase import Atoms
from ase.io import read
from gpaw import GPAW, PW
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from datetime import datetime

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUTS = REPO_ROOT / "data" / "inputs" / "VASP_inputs"
DATA_OUTPUTS = REPO_ROOT / "data" / "outputs"
GPAW_OUTPUTS = DATA_OUTPUTS / "gpaw_calculations"
H2_REFERENCE_FILE = DATA_OUTPUTS / "h2_reference_energy.json"
OUTPUT_CSV = DATA_OUTPUTS / "gpaw_h_adsorption_results_v2.csv"
OUTPUT_JSON = DATA_OUTPUTS / "gpaw_h_adsorption_results_v2.json"
ADSORBML_OUTPUT_CSV = DATA_OUTPUTS / "gpaw_adsorbml_results.csv"
ADSORBML_OUTPUT_JSON = DATA_OUTPUTS / "gpaw_adsorbml_results.json"

# ZPE+entropy correction for H* (matches scripts/adsorbml/3-extract_rank.py).
# Cancels when comparing DFT vs ML ΔG_H, but kept for absolute placement.
ENTROPY_CORRECTION = 0.24  # eV

# Configuration
# Matches the OC20 (RPBE) VASP reference that UMA's adsorption head
# (task_name="oc20", used in scripts/adsorbml/2-run_adsorbml.py) was trained on.
# Verified against the OC20 paper (arXiv:2010.09990) and the fairchem source
# Open-Catalyst-Dataset/ocdata/utils/vasp.py (VASP_FLAGS):
#   gga="RP" (RPBE), encut=350, ediffg=-0.03, no ISPIN (non-spin-polarized),
#   no LDIPOL/IDIPOL (no dipole correction), no ISMEAR/SIGMA set → VASP defaults
#   (Methfessel-Paxton order 1, sigma 0.2 eV), Monkhorst-Pack k-mesh round(40/|a|).
# Irreducible gap: GPAW ships its own PAW datasets and cannot load VASP's, so a
# residual ~tens-of-meV offset vs VASP remains.
GPAW_CONFIG = {
    'mode': 'pw',            # Plane waves (like VASP); replaces LCAO/DZP
    'ecut': 350,             # eV, OC20 ENCUT=350
    'xc': 'RPBE',            # OC20 gga="RP"
    'kpts': 'auto',          # OC20 Monkhorst-Pack density (see _oc20_kpts)
    'spinpol': False,        # OC20 is NOT spin-polarized (VASP ISPIN default 1)
    # OC20 leaves ISMEAR/SIGMA at VASP defaults: Methfessel-Paxton order 1, 0.2 eV.
    'smearing': {'name': 'methfessel-paxton', 'width': 0.2, 'order': 1},
    'txt': 'gpaw.txt',       # Output log file
    'convergence': {
        'energy': 1e-5,      # Energy convergence (eV) — tighter than OC20 EDIFF=1e-4
        'density': 1e-4,     # Electron density convergence
        'eigenstates': 1e-5, # Eigenstate convergence
    },
}

RELAXATION_CONFIG = {
    'fmax': 0.03,            # eV/Å, matches OC20 EDIFFG=-0.03 (was 0.10)
    'steps': 200,            # practical cap; OC20 allowed NSW=2000 (was 8)
}

USE_RELAXATION = True

# Parallelism: cores allocated per GPAW calculation
CORES_PER_CALC = 11
# RAM per calculation (GB) for auto-detection
RAM_PER_CALC_GB = 4
# Per-structure hard timeout (hours). 0 disables timeout.
MAX_HOURS_PER_STRUCTURE = 0.0

# CSV header for incremental writes
CSV_COLUMNS = [
    'formula', 'surface_facet', 'adsorbate',
    'E_clean_slab_eV', 'E_slab_with_h_eV', 'E_h2_eV',
    'E_ads_eV', 'ΔGH_eV', 'source',
    'adsorption_site', 'h2_source', 'relaxed',
    'status', 'timestamp',
]

# CSV header for AdsorbML-mode results (separate file, includes ML comparison column)
ADSORBML_CSV_COLUMNS = CSV_COLUMNS + ['gibbs_free_ml_eV']

# ── Graceful shutdown ────────────────────────────────────────────
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n⚠️  Shutdown requested — finishing current calculations, then exiting...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Auto-detect parallelism ─────────────────────────────────────
def _detect_max_workers(override_workers=None):
    """Determine how many parallel GPAW calculations to run."""
    if override_workers is not None:
        print(f"Using manual worker override: {override_workers}")
        return max(1, int(override_workers))

    cpu_count = os.cpu_count() or 4
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
            else:
                ram_gb = 16
    except OSError:
        ram_gb = 16

    by_cpu = max(1, cpu_count // CORES_PER_CALC)
    by_ram = max(1, int(ram_gb // RAM_PER_CALC_GB))
    workers = min(by_cpu, by_ram)
    print(f"Auto-detected: {cpu_count} CPUs, {ram_gb:.0f} GB RAM → {workers} parallel workers")
    return workers


def _set_thread_env(threads_per_calc):
    """Set thread environment for BLAS/OpenMP libraries."""
    t = max(1, int(threads_per_calc))
    os.environ['OMP_NUM_THREADS'] = str(t)
    os.environ['OPENBLAS_NUM_THREADS'] = str(t)
    os.environ['MKL_NUM_THREADS'] = str(t)
    os.environ['NUMEXPR_NUM_THREADS'] = str(t)
    print(f"Thread env: OMP_NUM_THREADS={t}, OPENBLAS_NUM_THREADS={t}, MKL_NUM_THREADS={t}")


# ── Incremental CSV ──────────────────────────────────────────────
def _ensure_csv_header(csv_path, columns=None):
    """Create CSV with header if it doesn't exist."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns if columns is not None else CSV_COLUMNS)


def _append_result_csv(result_row, csv_path):
    """Append a single result row to CSV with file locking."""
    csv_path = Path(csv_path)
    with open(csv_path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow(result_row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _load_completed_keys(csv_path):
    """Load set of (formula, surface_facet) already completed in CSV."""
    csv_path = Path(csv_path)
    completed = set()
    if not csv_path.exists():
        return completed
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if row.get('status') == 'completed':
                completed.add((str(row['formula']), str(row['surface_facet'])))
    except Exception:
        pass
    return completed


def _make_row(formula, surface, *, e_clean=None, e_with_h=None, e_h2=None,
              e_ads=None, dgh=None, source=None, site=None, h2_source=None,
              relaxed=None, status='failed', timestamp=None):
    """Assemble one result row in CSV_COLUMNS order.

    `source` defaults to GPAW_<xc>; `relaxed` defaults to the current USE_RELAXATION;
    `timestamp` defaults to now. AdsorbML callers append the extra ML column themselves.
    """
    return [
        formula,
        surface,
        'H',
        e_clean,
        e_with_h,
        e_h2,
        e_ads,
        dgh,
        source if source is not None else f'GPAW_{GPAW_CONFIG["xc"]}',
        site,
        h2_source,
        bool(USE_RELAXATION) if relaxed is None else relaxed,
        status,
        timestamp if timestamp is not None else datetime.now().isoformat(),
    ]


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


def filter_structures_by_name(structures, selected_names):
    """Keep only structure directories whose names were explicitly requested."""
    if not selected_names:
        return structures

    requested = [name.strip() for name in selected_names if name and name.strip()]
    if not requested:
        return structures

    requested_set = set(requested)
    filtered = [item for item in structures if item[2].name in requested_set]
    missing = [name for name in requested if name not in {item[2].name for item in filtered}]
    if missing:
        raise ValueError(f"Requested structures not found: {', '.join(missing)}")
    return filtered


def setup_gpaw_calculator(label='gpaw', atoms=None):
    """
    Create GPAW calculator matching the OC20/RPBE VASP reference.

    Args:
        label: Label for calculation files
        atoms: Atoms object, required when kpts='auto' (OC20 per-cell k-mesh)

    Returns:
        GPAW calculator object
    """
    kpts = GPAW_CONFIG['kpts']
    if kpts == 'auto':
        if atoms is None:
            raise ValueError("kpts='auto' requires `atoms` to compute the OC20 k-mesh")
        kpts = _oc20_kpts(atoms)
    return GPAW(
        mode=PW(GPAW_CONFIG['ecut']),
        xc=GPAW_CONFIG['xc'],
        kpts=kpts,
        spinpol=GPAW_CONFIG['spinpol'],
        occupations=dict(GPAW_CONFIG['smearing']),
        txt=label + '.txt',
        convergence=GPAW_CONFIG['convergence'],
    )


def _oc20_kpts(atoms):
    """OC20 Monkhorst-Pack mesh: round(40/|a_i|) in-plane, 1 out-of-plane.

    Matches Open-Catalyst-Dataset/ocdata/utils/vasp.py:calculate_surface_k_points
    (multiplier 40, infinity-norm of the first two cell vectors).
    """
    cell = atoms.get_cell()
    a = np.linalg.norm(cell[0], ord=np.inf)
    b = np.linalg.norm(cell[1], ord=np.inf)
    return (max(1, int(round(40 / a))), max(1, int(round(40 / b))), 1)


def _prepare_slab(slab_file):
    """Load slab and enforce minimum vacuum (repeated-slab, 3D-periodic like OC20)."""
    slab = read(slab_file)
    if slab.cell[2, 2] < 10:
        slab.cell[2, 2] = 15
        slab.center(axis=2, vacuum=0)
    return slab


def _maybe_relax(atoms, label):
    """Optional local relaxation for improved adsorption geometry."""
    if not USE_RELAXATION:
        return atoms

    # Ensure FixAtoms constraints survived .copy()
    if not atoms.constraints:
        positions = atoms.get_positions()
        z = positions[:, 2]
        z_mid = (np.min(z) + np.max(z)) / 2
        fixed = [i for i in range(len(atoms)) if atoms[i].z < z_mid]
        if fixed:
            atoms.set_constraint(FixAtoms(indices=fixed))

    optimizer = BFGS(atoms, logfile=f"{label}_relax.log")
    optimizer.run(fmax=RELAXATION_CONFIG['fmax'], steps=RELAXATION_CONFIG['steps'])
    return atoms


def _slab_energy(atoms, label, relax):
    """Attach a GPAW calculator to a copy of `atoms`, optionally relax, return energy (eV)."""
    atoms = atoms.copy()
    atoms.calc = setup_gpaw_calculator(label=label, atoms=atoms)
    if relax:
        atoms = _maybe_relax(atoms, label)
    return atoms.get_potential_energy()


def _delta_gh(e_with_h, e_clean, e_h2):
    """Return (E_ads, ΔGH). E_ads is raw electronic; ΔGH adds the ZPE/entropy correction
    so it is directly comparable to the ML gibbs_free_ml_eV (which also includes it)."""
    e_ads = e_with_h - e_clean - 0.5 * e_h2
    return e_ads, e_ads + ENTROPY_CORRECTION


def _candidate_h_positions(slab, h_distance=1.5, max_sites=6):
    """Generate adsorption candidates from top-layer atomic positions."""
    positions = slab.get_positions()
    top_z = np.max(positions[:, 2])
    z_tol = 0.35

    top_indices = [i for i, pos in enumerate(positions) if (top_z - pos[2]) <= z_tol]
    if not top_indices:
        center_xy = np.mean(positions[:, :2], axis=0)
        return [("center_fallback", [center_xy[0], center_xy[1], top_z + h_distance])]

    top_xy = positions[top_indices, :2]
    center_xy = np.mean(top_xy, axis=0)
    distances = [np.linalg.norm(xy - center_xy) for xy in top_xy]
    ranked = [xy for _, xy in sorted(zip(distances, top_xy), key=lambda x: x[0])]

    candidates = []
    for i, xy in enumerate(ranked[:max_sites]):
        candidates.append((f"top_{i+1}", [float(xy[0]), float(xy[1]), float(top_z + h_distance)]))

    # Add one bridge candidate between two most central top atoms when available.
    if len(ranked) >= 2:
        bridge_xy = 0.5 * (ranked[0] + ranked[1])
        candidates.append(("bridge_center", [float(bridge_xy[0]), float(bridge_xy[1]), float(top_z + h_distance)]))

    # Add a centroid candidate for broad coverage.
    candidates.append(("top_centroid", [float(center_xy[0]), float(center_xy[1]), float(top_z + h_distance)]))

    unique = []
    for label, pos in candidates:
        duplicate = any(np.allclose(pos[:2], other_pos[:2], atol=1e-3) for _, other_pos in unique)
        if not duplicate:
            unique.append((label, pos))
    return unique


def calculate_clean_slab_energy(slab, output_dir):
    """
    Calculate energy of clean surface (no adsorbate)
    
    Args:
        slab: ASE Atoms object (already prepared)
        output_dir: Directory to save results
        
    Returns:
        float: Total energy in eV, or None if failed
    """
    
    try:
        print(f"  └─ Calculating clean slab energy...")
        energy = _slab_energy(slab, f'{output_dir}/clean_slab', relax=USE_RELAXATION)
        print(f"     ✓ Clean slab: E = {energy:.6f} eV")
        return energy

    except Exception as e:
        print(f"     ✗ Error: {e}")
        return None


def calculate_slab_with_h_energy(slab, output_dir, h_distance=1.5):
    """
    Calculate energy of surface with H adsorbate
    
    Args:
        slab: ASE Atoms object (already prepared)
        output_dir: Directory to save results
        h_distance: Distance of H above surface (Å)
        
    Returns:
        tuple: (best_energy, best_site_label, best_position) or (None, None, None)
    """
    
    try:
        print(f"  └─ Calculating slab+H energy...")
        
        candidates = _candidate_h_positions(slab, h_distance=h_distance)

        best_energy = None
        best_site = None
        best_position = None

        for site_label, h_pos in candidates:
            slab_with_h = slab.copy()
            slab_with_h += Atoms('H', positions=[h_pos])

            energy = _slab_energy(slab_with_h, f'{output_dir}/slab_with_h_{site_label}',
                                  relax=USE_RELAXATION)
            print(f"     · site={site_label:<12} E = {energy:.6f} eV")

            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_site = site_label
                best_position = h_pos

        print(f"     ✓ Best slab+H: site={best_site}, E = {best_energy:.6f} eV")
        return best_energy, best_site, best_position
    
    except Exception as e:
        print(f"     ✗ Error: {e}")
        return None, None, None


def calculate_h2_molecule_energy(output_dir):
    """
    Calculate energy of isolated H2 molecule
    
    Note: This is expensive, so we use a large vacuum and simpler settings
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        float: Energy of H2 molecule (eV), or None if failed
    """
    
    try:
        print(f"  └─ Calculating H₂ molecule energy...")
        
        # Create H2 molecule in a large box
        h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.75]])

        # Add vacuum; PW mode requires periodic boundary conditions
        h2.center(vacuum=10)
        h2.pbc = True

        # Setup GPAW at the same level of theory as the slabs. No dipole
        # correction (H2 is non-polar and this is a box, not a slab); Γ-point only.
        calc = GPAW(
            mode=PW(GPAW_CONFIG['ecut']),
            xc=GPAW_CONFIG['xc'],
            kpts=(1, 1, 1),  # Only 1 k-point (Γ) for isolated molecule
            spinpol=GPAW_CONFIG['spinpol'],
            occupations=dict(GPAW_CONFIG['smearing']),
            txt=f'{output_dir}/h2_molecule.txt',
            convergence=GPAW_CONFIG['convergence'],
        )
        
        h2.calc = calc
        energy = h2.get_potential_energy()
        print(f"     ✓ H₂ molecule: E = {energy:.6f} eV")
        
        return energy
    
    except Exception as e:
        print(f"     ✗ Error: {e}")
        return None


def get_h2_reference_energy():
    """Get H2 reference energy from cache or compute once for this campaign."""
    if H2_REFERENCE_FILE.exists():
        try:
            with open(H2_REFERENCE_FILE, 'r') as f:
                payload = json.load(f)
            energy = float(payload['E_h2_eV'])
            print(f"✓ Loaded cached H2 reference: {energy:.6f} eV")
            return energy, "cache"
        except Exception as exc:
            print(f"⚠️  Ignoring invalid H2 cache ({exc}), recomputing...")

    h2_dir = GPAW_OUTPUTS / "h2_reference"
    h2_dir.mkdir(parents=True, exist_ok=True)
    energy = calculate_h2_molecule_energy(str(h2_dir))
    if energy is None:
        raise RuntimeError("Failed to compute H2 reference energy; aborting run")

    H2_REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(H2_REFERENCE_FILE, 'w') as f:
        json.dump(
            {
                'E_h2_eV': float(energy),
                'timestamp': datetime.now().isoformat(),
                'gpaw_config': GPAW_CONFIG,
                'relaxation_steps': RELAXATION_CONFIG['steps'],
                'use_relaxation': USE_RELAXATION,
            },
            f,
            indent=2,
        )
    print(f"✓ Saved H2 reference cache: {H2_REFERENCE_FILE}")
    return energy, "computed"


def calculate_surface_properties(formula, miller, slab_file, output_dir, e_h2, h2_source):
    """
    Calculate all energies and ΔGH for a surface.

    Loads the slab once and passes it to both clean and H calculations.
    Writes result to CSV incrementally.
    """
    
    print(f"\n{'='*60}")
    print(f"Calculating {formula} {miller}")
    print(f"{'='*60}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    result = {
        'formula': formula,
        'surface': miller,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending',
        'E_clean_slab': None,
        'E_slab_with_h': None,
        'E_h2': None,
        'E_ads': None,
        'ΔGH': None,
        'adsorption_site': None,
        'h_position': None,
        'h2_source': h2_source,
        'relaxed': bool(USE_RELAXATION),
        'error': None,
    }
    
    try:
        # Check if file exists
        if not os.path.exists(slab_file):
            raise FileNotFoundError(f"POSCAR not found: {slab_file}")

        # Load slab ONCE
        slab = _prepare_slab(slab_file)
        
        # Calculate clean slab
        e_clean = calculate_clean_slab_energy(slab, output_dir)
        if e_clean is None:
            raise RuntimeError("Failed to calculate clean slab energy")
        result['E_clean_slab'] = float(e_clean)
        
        # Calculate slab with H (pass same slab object)
        e_with_h, best_site, best_position = calculate_slab_with_h_energy(slab, output_dir)
        if e_with_h is None:
            raise RuntimeError("Failed to calculate slab+H energy")
        result['E_slab_with_h'] = float(e_with_h)
        result['adsorption_site'] = best_site
        result['h_position'] = best_position
        
        result['E_h2'] = float(e_h2)
        
        # Calculate E_ads (raw electronic) and ΔGH (free energy)
        e_ads, delta_gh = _delta_gh(e_with_h, e_clean, e_h2)
        result['E_ads'] = float(e_ads)
        result['ΔGH'] = float(delta_gh)
        result['status'] = 'completed'

        print(f"\n  Results for {formula} {miller}:")
        print(f"    E(clean slab) = {e_clean:.6f} eV")
        print(f"    E(slab+H)     = {e_with_h:.6f} eV")
        print(f"    E(H₂)         = {e_h2:.6f} eV")
        print(f"    E_ads         = {e_ads:.6f} eV")
        print(f"    ΔGH           = {delta_gh:.6f} eV ← KEY RESULT!")
        
        if -0.2 < delta_gh < 0.2:
            print(f"    ✓ EXCELLENT! Near optimal for HER")
        elif -0.5 < delta_gh < 0.5:
            print(f"    ○ GOOD, reasonable for HER")
        else:
            print(f"    ⚠️  Outside typical HER range")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"  ✗ Error: {e}")

    # Write result to CSV immediately
    row = _make_row(
        result['formula'], result['surface'],
        e_clean=result['E_clean_slab'], e_with_h=result['E_slab_with_h'],
        e_h2=result['E_h2'], e_ads=result['E_ads'], dgh=result['ΔGH'],
        site=result.get('adsorption_site'), h2_source=result.get('h2_source'),
        relaxed=result.get('relaxed'), status=result['status'],
        timestamp=result['timestamp'],
    )
    _append_result_csv(row, OUTPUT_CSV)

    return result


def _write_failed_row(formula, surface, h2_source, reason):
    """Write a failed result row directly (used for timeout/worker failures)."""
    row = _make_row(formula, surface, h2_source=h2_source, status='failed')
    _append_result_csv(row, OUTPUT_CSV)
    print(f"  ✗ Marked failed: {formula} {surface} ({reason})")


def _timeout_handler(signum, frame):
    raise TimeoutError("Per-structure timeout reached")


def _compute_one(args):
    """Worker function for parallel execution."""
    formula, surface, poscar_dir, e_h2, h2_source, max_seconds = args
    poscar_file = poscar_dir / "POSCAR"
    output_dir = GPAW_OUTPUTS / poscar_dir.name

    if max_seconds and max_seconds > 0:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(max_seconds))

    t0 = time.time()
    try:
        return calculate_surface_properties(
            formula=formula,
            miller=surface,
            slab_file=str(poscar_file),
            output_dir=str(output_dir),
            e_h2=e_h2,
            h2_source=h2_source,
        )
    except TimeoutError as exc:
        _write_failed_row(formula, surface, h2_source, str(exc))
        return {
            'formula': formula,
            'surface': surface,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(exc),
        }
    finally:
        elapsed_min = (time.time() - t0) / 60.0
        print(f"⏱ Finished worker task {formula} {surface} in {elapsed_min:.1f} min")
        if max_seconds and max_seconds > 0:
            signal.alarm(0)


def _run_pool(pending, worker_fn, workers_override=None, label_fn=repr):
    """Run `worker_fn` over `pending` args, sequentially or via ProcessPoolExecutor.

    Honors the global shutdown flag and logs per-task exceptions using
    `label_fn(args)`. Returns the list of worker return values.
    """
    max_workers = _detect_max_workers(override_workers=workers_override)
    results = []

    if max_workers <= 1:
        for args in pending:
            if _shutdown_requested:
                print("⚠️  Shutdown: stopping before next task")
                break
            results.append(worker_fn(args))
        return results

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker_fn, args): args for args in pending}
        for future in as_completed(future_map):
            if _shutdown_requested:
                print("⚠️  Shutdown: cancelling remaining futures")
                executor.shutdown(wait=False, cancel_futures=True)
                break
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"  ✗ Worker exception for {label_fn(future_map[future])}: {exc}")
    return results


# ── AdsorbML mode ───────────────────────────────────────────────
def _load_adsorbml_structures(csv_path):
    """Read ranked_candidates.csv and return list of dicts for _compute_one_adsorbml."""
    df = pd.read_csv(csv_path)
    required = {'slab_name', 'slab_file', 'candidate_file', 'gibbs_free_ml_eV'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ranked_candidates.csv missing columns: {missing}")
    return df.to_dict('records')


def _compute_one_adsorbml(args):
    """Worker for AdsorbML mode: single-point DFT on a pre-relaxed slab + adslab."""
    row, e_h2, h2_source = args
    slab_name    = row['slab_name']
    slab_file    = row['slab_file']
    adslab_file  = row['candidate_file']
    gibbs_ml     = row['gibbs_free_ml_eV']

    output_dir = str(GPAW_OUTPUTS / slab_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AdsorbML DFT validation: {slab_name}")
    print(f"  ML ΔG*H = {gibbs_ml:.4f} eV")
    print(f"{'='*60}")

    timestamp = datetime.now().isoformat()
    status = 'failed'
    e_clean = e_with_h = e_ads = dgh = None

    try:
        clean_slab = _prepare_slab(slab_file)
        e_clean = _slab_energy(clean_slab, f'{output_dir}/clean_slab', relax=False)
        print(f"     ✓ Clean slab: E = {e_clean:.6f} eV")

        adslab = _prepare_slab(adslab_file)
        e_with_h = _slab_energy(adslab, f'{output_dir}/adslab', relax=False)
        print(f"     ✓ Adslab: E = {e_with_h:.6f} eV")

        e_ads, dgh = _delta_gh(e_with_h, e_clean, e_h2)
        status = 'completed'
        print(f"  E_ads (DFT) = {e_ads:.6f} eV   ΔGH (DFT) = {dgh:.6f} eV   ML ΔG*H = {gibbs_ml:.4f} eV")

    except Exception as exc:
        print(f"  ✗ Error: {exc}")

    row_data = _make_row(
        slab_name, slab_name,
        e_clean=e_clean, e_with_h=e_with_h, e_h2=e_h2, e_ads=e_ads, dgh=dgh,
        source=f'GPAW_{GPAW_CONFIG["xc"]}_adsorbml', site='adsorbml_best',
        h2_source=h2_source, relaxed=False, status=status, timestamp=timestamp,
    ) + [gibbs_ml]
    _append_result_csv(row_data, ADSORBML_OUTPUT_CSV)
    return {'formula': slab_name, 'surface': slab_name, 'status': status,
            'E_ads': e_ads, 'ΔGH': dgh, 'gibbs_free_ml_eV': gibbs_ml, 'timestamp': timestamp}


def run_adsorbml_calculations(candidates_csv, workers_override=None, selected_names=None):
    """Run single-point GPAW on AdsorbML pre-relaxed structures."""
    print("\n" + "="*60)
    print("GPAW AdsorbML Validation Mode")
    print("="*60)
    print(f"Starting time: {datetime.now()}")
    print(f"Candidates CSV: {candidates_csv}")

    _ensure_csv_header(ADSORBML_OUTPUT_CSV, columns=ADSORBML_CSV_COLUMNS)

    completed = _load_completed_keys(ADSORBML_OUTPUT_CSV)
    if completed:
        print(f"✓ Checkpoint: {len(completed)} structures already completed, will skip")

    e_h2, h2_source = get_h2_reference_energy()

    structures = _load_adsorbml_structures(candidates_csv)
    if selected_names:
        selected_set = set(selected_names)
        structures = [s for s in structures if s['slab_name'] in selected_set]

    pending = [
        (row, e_h2, h2_source)
        for row in structures
        if (row['slab_name'], row['slab_name']) not in completed
    ]
    print(f"Structures to compute: {len(pending)} (skipped {len(structures) - len(pending)} completed)")

    if not pending:
        print("✓ All structures already completed!")
        return []

    return _run_pool(pending, _compute_one_adsorbml,
                     workers_override=workers_override,
                     label_fn=lambda a: a[0]['slab_name'])


def run_calculations_parallel(base_dir=None,
                             include_patterns=None,
                             selected_structure_names=None,
                             workers_override=None,
                             max_hours_per_structure=None):
    """
    Discover POSCARs and run all calculations with parallel workers and checkpoint/resume.

    Args:
        include_patterns: optional list of glob patterns to filter structures
    """

    print("\n" + "="*60)
    print("GPAW H Adsorption Energy Calculator (v2)")
    print("="*60)
    print(f"\nStarting time: {datetime.now()}")
    print(f"Relaxation: {'ON (' + str(RELAXATION_CONFIG['steps']) + ' steps)' if USE_RELAXATION else 'OFF'}")
    if include_patterns:
        print(f"Include filter: {', '.join(include_patterns)}")
    if max_hours_per_structure and max_hours_per_structure > 0:
        print(f"Per-structure timeout: {max_hours_per_structure:.2f} hours")

    base_dir = Path(base_dir) if base_dir else DATA_INPUTS

    # Ensure output CSV exists with header
    _ensure_csv_header(OUTPUT_CSV)

    # Load checkpoint: skip already-completed structures
    completed = _load_completed_keys(OUTPUT_CSV)
    if completed:
        print(f"✓ Checkpoint: {len(completed)} structures already completed, will skip")

    e_h2, h2_source = get_h2_reference_energy()

    structures = discover_structures(base_dir, include_patterns=include_patterns)
    structures = filter_structures_by_name(structures, selected_structure_names)
    if not structures:
        print("\n⚠️  No POSCAR files found in data/inputs/VASP_inputs")
        return []

    max_seconds = 0
    if max_hours_per_structure and max_hours_per_structure > 0:
        max_seconds = int(max_hours_per_structure * 3600)
    pending = [
        (formula, surface, poscar_dir, e_h2, h2_source, max_seconds)
        for formula, surface, poscar_dir in structures
        if (formula, surface) not in completed
    ]

    print(f"Structures to compute: {len(pending)} (skipped {len(structures) - len(pending)} completed)")

    if not pending:
        print("✓ All structures already completed!")
        return []

    return _run_pool(pending, _compute_one, workers_override=workers_override,
                     label_fn=lambda a: f"{a[0]} {a[1]}")


def save_results(results, json_file=None,
                csv_file=None):
    """
    Save summary JSON and print statistics.

    Note: CSV is written incrementally during computation.
    This function saves the JSON backup and prints a summary.
    """
    
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    json_file = Path(json_file) if json_file else OUTPUT_JSON
    csv_file = Path(csv_file) if csv_file else OUTPUT_CSV

    # Save JSON (full details)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Full results saved to: {json_file}")
    print(f"✓ Incremental CSV at: {csv_file}")
    
    # Print summary from the CSV (which has all results including previous runs)
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        df = pd.DataFrame()

    if len(df) > 0:
        completed = df[df['status'] == 'completed']
        print(f"\nTotal entries in CSV: {len(df)}")
        print(f"Successfully calculated: {len(completed)} surfaces")

        if len(completed) > 0:
            print("\nΔGH statistics by formula:")
            for formula in sorted(completed['formula'].unique()):
                formula_df = completed[completed['formula'] == formula]
                dgh_values = formula_df['ΔGH_eV'].dropna().values
                if len(dgh_values) > 0:
                    print(f"  {formula}:")
                    print(f"    Count: {len(dgh_values)}")
                    print(f"    Mean ΔGH: {np.mean(dgh_values):.4f} eV")
                    print(f"    Min ΔGH:  {np.min(dgh_values):.4f} eV (best surface)")
                    print(f"    Max ΔGH:  {np.max(dgh_values):.4f} eV")

            # Highlight best candidates
            excellent = completed[completed['ΔGH_eV'].abs() < 0.2]
            if len(excellent) > 0:
                print(f"\n✓ EXCELLENT candidates (|ΔGH| < 0.2 eV): {len(excellent)}")
                print(excellent[['formula', 'surface_facet', 'ΔGH_eV']].to_string(index=False))

        failed = df[df['status'] == 'failed']
        if len(failed) > 0:
            print(f"\n⚠️  Failed calculations: {len(failed)}")
            print(failed[['formula', 'surface_facet', 'status']].to_string(index=False))
    
    print(f"\nEnd time: {datetime.now()}")


# ── Pre-defined machine splits ───────────────────────────────────
MACHINE_SPLITS = {
    # DEVANA (64 CPUs): 159 structures (~46.5%)
    'devana': [
        'Ni_Mo2N_interface_*',
        'Ni_MoS2_interface_*',
        'Mo2N_*',
        'MoS2_*',
        'MoSe2_*',
        'MoP_*',
        'graphene_*',
        'Ni2_on_*',
        'Ni4_on_*',
    ],
    # NODE1 (32 CPUs): 79 structures (~23.1%)
    'node1': [
        'Ni_MoB_interface_*',
        'MoB_*',
    ],
    # NODE2 (24 CPUs): 61 structures (~17.8%)
    'node2': [
        'Ni_Mo2C_interface_*',
        'Ni_Ti3C2O2_interface_*',
        'Ti3C2O2_*',
    ],
    # NODE3 (16 CPUs): 43 structures (~12.6%)
    'node3': [
        'Mo2C_*',
    ],
}


def main():
    """Main execution"""

    global USE_RELAXATION, CORES_PER_CALC

    parser = argparse.ArgumentParser(
        description='GPAW H Adsorption Calculator (v2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Machine splits (use --machine instead of --include):\n'
            + '\n'.join(f'  {k:14s} → {" , ".join(v)}' for k, v in MACHINE_SPLITS.items())
        ),
    )
    parser.add_argument(
        '--include', type=str, default=None,
        help='Comma-separated glob patterns to filter structures, '
             'e.g. "Ni_Mo2C_interface_*,Ni_MoB_interface_*"',
    )
    parser.add_argument(
        '--machine', type=str, default=None,
        choices=list(MACHINE_SPLITS.keys()),
        help='Use a pre-defined split for a specific machine',
    )
    parser.add_argument(
        '--list', action='store_true', dest='list_only',
        help='Only list structures that would be computed, then exit',
    )
    parser.add_argument(
        '--write-structure-list', type=str, default=None,
        help='Write matched structure directory names to a text file, one per line, then exit',
    )
    parser.add_argument(
        '--structure-name', action='append', default=None,
        help='Run only the given structure directory name. May be provided multiple times or as a comma-separated list.',
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Manual number of parallel workers (overrides auto-detection)',
    )
    parser.add_argument(
        '--cores-per-calc', type=int, default=None,
        help='CPU cores budget per calculation used by auto worker detection',
    )
    parser.add_argument(
        '--relax-steps', type=int, default=None,
        help='Override relaxation steps (default from config)',
    )
    parser.add_argument(
        '--fmax', type=float, default=None,
        help='Override BFGS force threshold in eV/A',
    )
    parser.add_argument(
        '--no-relax', action='store_true',
        help='Disable structural relaxation (single-point energies only)',
    )
    parser.add_argument(
        '--kpts', type=str, default=None,
        help='Override k-point mesh as comma-separated triple, e.g. 2,2,1',
    )
    parser.add_argument(
        '--max-hours-per-structure', type=float, default=0.0,
        help='Hard timeout per structure in hours (0 disables timeout)',
    )
    parser.add_argument(
        '--adsorbml-candidates', type=str, default=None,
        metavar='PATH',
        help='Path to ranked_candidates.csv from scripts/adsorbml/3-extract_rank.py. '
             'When provided, skips POSCAR discovery and H-site search and instead '
             'runs single-point DFT on AdsorbML pre-relaxed structures. '
             'Results are written to data/outputs/gpaw_adsorbml_results.csv.',
    )
    args = parser.parse_args()

    # Resolve include patterns
    include_patterns = None
    if args.machine:
        include_patterns = MACHINE_SPLITS[args.machine]
        print(f"Using machine split: {args.machine}")
    elif args.include:
        include_patterns = [p.strip() for p in args.include.split(',')]

    selected_structure_names = None
    if args.structure_name:
        selected_structure_names = []
        for value in args.structure_name:
            selected_structure_names.extend(part.strip() for part in value.split(',') if part.strip())
        print(f"Explicit structures: {', '.join(selected_structure_names)}")

    # Runtime overrides
    if args.cores_per_calc is not None:
        CORES_PER_CALC = max(1, int(args.cores_per_calc))
        print(f"Override: CORES_PER_CALC={CORES_PER_CALC}")

    _set_thread_env(CORES_PER_CALC)

    if args.no_relax:
        USE_RELAXATION = False
        print("Override: relaxation disabled")

    if args.relax_steps is not None:
        RELAXATION_CONFIG['steps'] = max(0, int(args.relax_steps))
        print(f"Override: relax_steps={RELAXATION_CONFIG['steps']}")

    if args.fmax is not None:
        RELAXATION_CONFIG['fmax'] = float(args.fmax)
        print(f"Override: fmax={RELAXATION_CONFIG['fmax']}")

    if args.kpts:
        try:
            parts = tuple(int(x.strip()) for x in args.kpts.split(','))
            if len(parts) != 3:
                raise ValueError("kpts must have 3 integers")
            GPAW_CONFIG['kpts'] = parts
            print(f"Override: kpts={GPAW_CONFIG['kpts']}")
        except Exception as exc:
            raise ValueError(f"Invalid --kpts '{args.kpts}': {exc}")

    # List-only mode
    if args.list_only or args.write_structure_list:
        structures = discover_structures(DATA_INPUTS, include_patterns=include_patterns)
        structures = filter_structures_by_name(structures, selected_structure_names)
        print(f"\nStructures matched: {len(structures)}")
        for formula, surface, poscar_dir in structures:
            print(f"  {poscar_dir.name}")
        if args.write_structure_list:
            output_path = Path(args.write_structure_list)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(''.join(f"{poscar_dir.name}\n" for _, _, poscar_dir in structures))
            print(f"\n✓ Wrote structure manifest: {output_path}")
        if args.list_only or args.write_structure_list:
            return

    if selected_structure_names:
        args.workers = 1
        print("Single-structure mode: forcing workers=1 for scheduler-friendly execution")

    # Invalidate H2 cache if relaxation config or XC functional changed
    if H2_REFERENCE_FILE.exists():
        try:
            with open(H2_REFERENCE_FILE, 'r') as f:
                cache = json.load(f)
            cached_steps = cache.get('relaxation_steps')
            cached_cfg = cache.get('gpaw_config', {})
            cached_xc = cached_cfg.get('xc')
            cached_mode = cached_cfg.get('mode')
            cached_ecut = cached_cfg.get('ecut')
            if cached_steps != RELAXATION_CONFIG['steps']:
                print(f"⚠️  H2 cache mismatch (steps {cached_steps}→{RELAXATION_CONFIG['steps']}), recomputing")
                H2_REFERENCE_FILE.unlink()
            elif cached_xc != GPAW_CONFIG['xc']:
                print(f"⚠️  H2 cache mismatch (xc {cached_xc}→{GPAW_CONFIG['xc']}), recomputing")
                H2_REFERENCE_FILE.unlink()
            elif (cached_mode, cached_ecut) != (GPAW_CONFIG['mode'], GPAW_CONFIG['ecut']):
                print(f"⚠️  H2 cache mismatch (mode/ecut {cached_mode}/{cached_ecut}→"
                      f"{GPAW_CONFIG['mode']}/{GPAW_CONFIG['ecut']}), recomputing")
                H2_REFERENCE_FILE.unlink()
        except Exception:
            pass
    
    # ── AdsorbML validation mode ─────────────────────────────────
    if args.adsorbml_candidates:
        candidates_path = Path(args.adsorbml_candidates)
        if not candidates_path.exists():
            raise FileNotFoundError(f"--adsorbml-candidates file not found: {candidates_path}")
        results = run_adsorbml_calculations(
            candidates_csv=str(candidates_path),
            workers_override=args.workers,
            selected_names=selected_structure_names,
        )
        save_results(results, json_file=ADSORBML_OUTPUT_JSON, csv_file=ADSORBML_OUTPUT_CSV)
        print("\n✓ Done! AdsorbML DFT results saved to:")
        print(f"  - {ADSORBML_OUTPUT_CSV} (incremental, crash-safe)")
        print(f"  - {ADSORBML_OUTPUT_JSON} (JSON summary)")
        return

    # ── Standard fresh-POSCAR mode ───────────────────────────────
    results = run_calculations_parallel(
        include_patterns=include_patterns,
        selected_structure_names=selected_structure_names,
        workers_override=args.workers,
        max_hours_per_structure=args.max_hours_per_structure,
    )

    # Save JSON summary
    save_results(results)

    print("\n✓ Done! Results saved to:")
    print(f"  - {OUTPUT_CSV} (incremental, crash-safe)")
    print(f"  - {OUTPUT_JSON} (JSON summary)")
    print("\nNext step: Combine with OCx24 and train ML model!")


if __name__ == '__main__':
    main()
