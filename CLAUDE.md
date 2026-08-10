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
    tagging.py                #   surface tags (1=free, 0=frozen) — see "Surface tagging" below
    audit_tags.py             #   CPU-only tag audit over all inputs; gate before spending GPU time
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
  (`run_adsorbml` returns only anomaly-free candidates). A slab that ends with **zero**
  candidates now logs an ERROR with the anomaly tally (both per-slab and in the batch
  reduce) instead of quietly writing an empty CSV.

## Surface tagging (`scripts/adsorbml/tagging.py`)

Tags are assigned once, in step 1, and drive **three** things in fairchem:
which atoms relax (`FixAtoms(mask=tag==0)`), **where H is placed** (Delaunay mesh over
tag-1 atoms only), and **what `is_adsorbate_intercalated` rejects** (any placement whose
H neighbours a tag-0 atom).

Because of the third, a frozen atom that H can physically touch turns every placement
over it into a false rejection. The old `z > z_max - 2.0` rule did exactly that on
**219/350 inputs**; the `Ni_*_cluster*` interfaces worst, freeing as few as 5 atoms of 262
(the cluster apex sets `z_max`, so a 2 Å window leaves the whole substrate rigid). That is
a real defect — surfaces that cannot relax — and it is fixed.

**It is not what caused the zero-candidate failure.** All 48 Mo₂N structures returning
0/100 was the placement-wrap bug (next section); tagging and the wrap bug were found in
the same investigation and are easy to conflate. The evidence they are separate: the
frozen-but-reachable count is *anti*-correlated with rejections (Mo₂C(100) had **0** such
atoms and 21 % rejections; Mo₂C(111) had **18** and 0.8 %), and Mo₂N goes 0 → 100
candidates with the **old** tags once the cell is padded. Impact of the tagging fix alone
on ΔG_H is small but nonzero: Mo₂C(110) shifted +0.011 eV, Mo₂C(100) unchanged to 3 d.p.

`surface_tags()` tags 1 if ANY of: OC20 height window (Cartesian z, inclusive);
under-coordination above the COM (uses the slab's own interior when no bulk is given);
**an H-sized probe reaches the atom from +z**; or the atom is in the top 2 atomic layers
(z-clustering, so the freed fraction is independent of interlayer spacing). Stacks of ≤3
layers are freed entirely. It **raises** if any reachable atom would stay frozen.

Note upstream's `fairchem...slab.tag_surface_atoms` does *not* fix this — verified, it
frees the same 12/192 on Mo₂N(001). Reachability is one-sided (+z) on purpose: AdsorbML
never places the adsorbate below the slab, so the bottom face stays frozen as the bulk
proxy.

```
python scripts/adsorbml/audit_tags.py [--include "Mo2N_*"] [--compare]   # CPU, no GPU
```
Exit status 1 if any structure fails an invariant, so it can gate a submission script. It
checks reachability *and* the wrap margin (next section). Currently **0/342 fail**
(free fraction min 8.6 %, median 17.8 %; wrap margin min +2.0 Å, median +4.4 Å). Run it
after any change to the generator or the tag rule.

Caveats: the probe is one-sided, so it cannot see undercuts, side-exposed faces (edge
ribbons — excluded from the pipeline anyway), or H migrating sideways during relaxation.
Results produced before this rule landed used a fixed 2 Å window and are not directly
comparable; Mo₂C(110) tags changed, so the pre-existing Mo₂C results need re-running
before they can be merged with new ones.

## Placement wrap bug — why cells must be tall (fairchem, upstream)

`_get_scaled_normal` (`fairchem/data/oc/core/adsorbate_slab_config.py`) centres the
adsorption site at the cell centre and calls `wrap()`, commenting that this means it
"[doesn't] need to deal with pbc issues". With `pbc z = True` that is false:

1. Centring a top-surface site pushes the slab's underside below z=0, and `wrap()` brings
   it back in **above** the site.
2. The overlap solver lifts the adsorbate to clear those phantom atoms — ~20 Å for Mo₂N,
   past the top of the cell (H at z≈59 in a 46 Å cell).
3. `ocp_adslab_generator` (`fairchem/core/components/calculate/recipes/adsorbml.py`) then
   sets `atoms.pbc = True`, folding z≈59 back to z≈13 — **inside the slab**.
4. H now neighbours frozen atoms → `is_adsorbate_intercalated` → discarded. Every
   placement, before any relaxation.

**Trigger: `2·span > cell_z`** (span = `z.max()−z.min()`), because centring puts the slab
in `[cell_z/2 − span, cell_z/2]`, which only stays above z=0 when `cell_z ≥ 2·span`.
Verified against the pre-fix results, r = 0.974 across four families:

| family | 2·span − cell_z | rejected/100 |
|---|---|---|
| Mo₂C(110) | −2.4 (safe) | 0.2 |
| Mo₂C(111) | −4.5 (safe) | 0.8 |
| Mo₂C(100) | **+1.4** | 20.9 |
| Mo₂N(001) | **+14.0** | 100.0 |

`pbc = [True, True, False]` is not an option — `FAIRChemCalculator` raises `MixedPBCError`
on non-uniform PBC and the recipe relaxes this same slab object.

**Why OC20 never hits it:** its own convention is 20 Å vacuum on a ≥7 Å slab, so
`cell_z ≈ 27 ≫ 2·7`. The bug is latent upstream, which is why conforming to the sizing
convention below is the actual fix. Two defences, in order:

- `generate_structures.py:_ensure_z_clearance` sets
  `cell_z = max(span + 20, 2·span + 2)` at the end of **every** builder, and
  `_assert_sizing_invariants` fails the generator if any emitted structure violates it.
- `audit_tags.py` reports a `wrap-margin` column (`cell_z − 2·span`) and exits 1 below
  1 Å, catching hand-edited or externally supplied structures on CPU.
- `2-run_adsorbml.py` still pads the cell defensively at read time. On conforming inputs
  it is a no-op; leave it as the net. (Padding is free — MLIP cost scales with atom count,
  not cell volume; E_slab changed by 0.053 meV when this was measured.)

## Slab sizing convention (OC20)

Structures are sized to OC20 — the dataset UMA's `oc20` head was trained on, and whose DFT
settings `GPAW_CONFIG` already matches. The numbers are literal, from
`fairchem/data/oc/core/slab.py`:

```
SlabGenerator(min_slab_size=7.0, min_vacuum_size=20.0, lll_reduce=False,
              center_slab=True, primitive=True, max_normal_search=1)
get_slabs(tol=0.3, bonds=None, max_broken_bonds=0, symmetrize=False)
tile_atoms(min_ab=8.0)
```

Constants live at the top of `generate_structures.py`: `MIN_SLAB_THICKNESS 7.0`,
`MIN_VACUUM 20.0`, `MIN_AB 8.0`, `MIN_AB_DEFECT` (= `MIN_AB`), `MAX_ATOMS_TARGET 250`
(OC22's ceiling, warning only), `MAX_ATOMS_INTERFACE 600`.

**Two different "thicknesses" — do not conflate them.**
`_material_thickness()` = atom span + one interlayer spacing; this is what `min_slab_size`
means. `_atom_span()` = `z.max()−z.min()`; this is what the wrap invariant is about.
Mo₂N(001) has a 6.00 Å span but **8.00 Å of material** (4 planes, 2.00 Å apart), so it does
satisfy OC20 despite the span reading below 7.

`create_slab` cuts to a thickness **in Å** via pymatgen `SlabGenerator`. It previously
passed a layer count to `ase.build.surface`, whose `layers` counts *oriented-unit-cell
repeats*, not atomic planes — so `layers=4` meant 16 planes / 30 Å for Mo₂N(001) but 9
planes / 11.8 Å for Mo₂C(111), and 8 stacked monolayers (46 Å) for "MoS₂ basal plane".

⚠ **`create_slab` deliberately does NOT call `standardize_bulk`.** SpacegroupAnalyzer
standardization swaps Mo₂C's b and c axes (4.725, 6.022, 5.195 → 4.725, 5.195, 6.022),
which would silently redefine `Mo2C_(110)` as the plane we call (101). Miller indices are
interpreted in the basis of whatever `create_*_bulk()` returns. Do not "improve" this.

Non-slab families get explicit builders, because a thickness floor is wrong for them:
`create_tmd_basal_slab` (MoS₂/MoSe₂ — one monolayer, hexagonal `mx2` cell),
`create_mxene_basal_slab` (one Ti₃C₂O₂ sheet). `create_tmd_monolayer` is separate and stays
for edge ribbons, which need a rectangular cell. Interfaces pass thicknesses in **Å**
(`in_layers=False`, 7 Å film + 7 Å substrate); only `create_ni_mxene_interface` passes
layers, because one "layer" there is one intact O-Ti-C-Ti-C-Ti-O sheet.

Resulting sizes — every structure now has a wrap margin ≥ 5 Å:

| | before | after |
|---|---|---|
| pristine slabs (14) | 63–256 atoms, 8–49 Å span | **27–144 atoms, 3.2–11.9 Å** |
| vacancy/dopant slabs (117) | 430–576 atoms | **25–144 atoms** |
| `Ni_*_interface_*` (186) | 312–950 atoms | 272–512 atoms |
| `*_sheet` (8) | 384–768 atoms | **dropped** |

The `_sheet` family is gone: it was a 4×4 yardstick that reproduced the 2×2 value to
1–3 meV, i.e. it confirmed the smaller cell is converged and then cost 4× per structure.

Known limits of the current sizing:
- **`termination=0`** is the default and is arbitrary — no more so than the single cut
  `ase.build.surface` returned, but arbitrary. The list is `get_slabs()` results plus a
  flipped copy of each asymmetric slab (`_flip_slab_z`), so both faces are reachable:
  Mo₂C(100) has 2, MoP(001) 2, Mo₂C(110) 3, Mo₂C(111) 7, Mo₂N(001)/(100) 1.

  ⚠ **Two facets changed termination when the engine changed**, because `get_slabs` returns
  the anion face first where the old ase cut happened to land on metal:

  | | old (ase) | new (`termination=0`) | old surface is now |
  |---|---|---|---|
  | `Mo2C_(100)` | Mo-terminated | **C-terminated** | `termination=1` |
  | `MoP_(001)` | Mo-terminated | **P-terminated** | `termination=1` |

  Everything else keeps its top-layer species ratio. This matters: the Mo₂C(100)
  ΔG_H = −0.692 eV reference was measured on the **Mo**-terminated surface, so it is not a
  valid regression target for the current default. Deciding which face to screen (or
  screening both) is open work — ranking by relaxed `E_slab` is the principled route, and
  is valid because terminations of one facet share composition and atom count.
- **`MIN_AB_DEFECT = MIN_AB`** means Mo₂N(001) carries one dopant per 4 top-layer Mo — a
  doped surface more than an isolated dopant. Raise to 10.0 if that matters. This does
  *not* fix the separate finding that the best H site lands 5.5–8.6 Å from the dopant;
  that needs site generation restricted to a radius around the defect, in step 2.
- **7 Å is an OC20 (metals/alloys) convention** and Mo₂N/Mo₂C/MoB/MoP are compounds (OC22,
  the compound dataset, uses ≥8 Å, 12 Å vacuum, symmetric slabs, all atoms free). **Checked
  on Mo₂N(001) 2026-08-10 and it holds** — see "Thickness convergence" below. Not re-checked
  for Mo₂C / MoB / MoP.
- **MoB(111)** has no clean atomic layering (a z-clustering probe collapses its 8.8 Å into
  one "plane"), so "surface layer" is ill-defined there. Predates this change.

### Thickness convergence (Mo₂N(001), UMA, 2026-08-10)

`MIN_SLAB_THICKNESS = 7.0` is justified, not assumed. SlabGenerator quantises to whole
oriented-cell repeats, so Mo₂N(001) only offers 8 / 16 / 24 / 32 Å — a "7 / 10 / 14 Å"
ladder does not exist, 10 and 14 both land on 16 Å. All four run through AdsorbML steps 1–3,
100/100 candidates kept each, all converged:

| material thickness | planes | atoms | ΔG_H (eV) | Δ vs next | step-2 wall time |
|---|---|---|---|---|---|
| **8 Å** (the default) | 4 | 48 | **−0.0535** | 5.1 meV | 12 min |
| 16 Å | 8 | 96 | −0.0587 | 2.6 meV | 27 min |
| 24 Å | 12 | 144 | −0.0613 | 1.0 meV | 38 min |
| 32 Å (the old slab) | 16 | 192 | −0.0623 | — | 53 min |

Monotonic, each doubling roughly halving the residual → extrapolated limit ≈ −0.063 eV.
**8 Å errs by 9 meV against 32 Å at 4.6× less compute.** The binding site is *identical* at
every thickness (H +0.13…+0.17 Å above the topmost atom, 5-fold Mo at 2.07–2.42 Å, no N
neighbour — the ordered N-vacancy hollow), so only the depth of frozen bulk varies. Surface
tagging also frees exactly 24 atoms / 2 layers in all four, which is what makes it a clean test.

Useful side result: the 32 Å slab reproduces ΔG_H = **−0.062 eV**, matching the pre-resize
campaign value for Mo₂N(001) — the resize did not change the answer for the same geometry.

Caveats: UMA only, no DFT; pristine Mo₂N(001) only (a defect or dopant could couple to the
slab depth differently); and 8 Å is thin enough that the *bulk* interior is not yet
converged — the per-repeat slab energy increment is off by 0.41 eV at one repeat and settles
at −493.2098 eV from the third. That cancels in ΔG (an `E_slab` difference), which is why the
adsorption energy converges much faster than the total energy.

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
