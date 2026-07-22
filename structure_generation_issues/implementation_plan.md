# Implementation plan — fix structure generation (bases, faceting, defects, interfaces, calculator)

**Status:** plan only, not implemented. Written 2026-07-09. Consolidates the three
issue docs in this folder into one execution plan.
**Re-verified 2026-07-09** against code, re-run audits, spglib-checked cells, and
Materials Project: all line numbers, both audit root causes, and the crystallography
(P̄6m2 / P6₃/mmc / I4₁/amd, coordinations, mp-219 / mp-27953) confirmed exact.
Corrections from that verification are folded in below and marked **[amended]**.

## Context

`scripts/generate_structures.py` produces the POSCARs that seed the entire pipeline
(UMA relax → AdsorbML screen → rank → GPAW ΔG_H). Three documented defect classes
(`base_structure_polymorph_fix_plan.md`, `vacancy_dopant_and_methodology_fix_plan.md`,
`interface_overlap_fix_plan.md`) make large parts of the current 334 inputs physically
wrong or silently invalid:

1. **Wrong base polymorphs** — Mo₂N (headline material) built as a hand-placed
   body-centered cell (mislabeled "anti-perovskite"), MoP as CsCl/B2, MoS₂/MoSe₂ as
   single monolayers in a two-layer box. Every slab/defect/edge/interface sliced from
   these inherits the error; relaxation cannot repair a wrong polymorph.
2. **Silent no-op vacancy/dopant** — **71** `*_vac*`/`*_dop*` POSCARs are byte-identical
   to their pristine parent (re-verified 2026-07-09): 8 vacancies + 55 bug-caused dopants
   + 8 interface `dopNi` that are identical by definition (Ni→Ni substitution). The bug:
   the top-layer selector anchors to the whole-slab `z_max` and `return slab` on no
   match, with no warning. Plus methodology gaps (defect coverage too high at 2×2;
   missing dipole correction — see WS3-B4 for the corrected vacuum picture).
3. **Interface atom overlaps** — 54 `Ni_*_interface_*` glue two mismatched in-plane
   lattices then stamp `set_cell(ni.cell)`, wrapping distinct atoms to 0.22–0.43 Å.

They **interact**: all edit the same file; the interface builder is shared between
#3 and the #2 "dope topmost Mo" fix; and "dope topmost Mo" is only meaningful once
the base (#1) and overlap (#3) are correct. So the *code* lands in one branch in a
fixed order, and the expensive *regenerate → re-relax → re-rank* runs **once** at the
end. Intended outcome: a generator that emits only physically valid, correctly-typed
structures, with a validation gate that makes future silent failures impossible.

## Decisions locked (from user + Fable 5 consult)

- **Mo₂N model:** β-Mo₂N **anti-anatase, I4₁/amd (No. 141)**, matching MP **mp-27953**
  — the experimental/DFT ground-state ordering; every N is octahedral NMo₆, every Mo
  has exactly 3 N (homogeneous half-fill). Build the 12-atom conventional (a≈4.20 Å,
  c≈8.0 Å; ideal rock-salt-derived a₀≈4.16, c=2a₀ also fine — cell relaxes to c/a≈1.89).
  Keep a **layered ordering** (alternate (001) N planes filled/empty) as a cheap
  bulk+ΔG_H sensitivity check; if ΔG_H shifts ≲0.1 eV, ordering is not ranking-limiting.
- **Interfaces:** **Option B — pymatgen ZSL / CoherentInterfaceBuilder** lattice
  matching (<2% strain), replacing the broken concatenation. Bigger cells / changed
  atom counts accepted.
- **Methodology:** **adopt** B1 larger defect supercell **and** the GPAW dipole
  correction. **[amended]** The B4 vacuum increase is dropped as unnecessary: ASE's
  `vacuum=8` is *per side*, so current slabs already have a **16 Å inter-image gap**
  (measured on the POSCARs) — above the ≥12 Å standard the original decision assumed
  was violated. Raising the parameter to 12–15 would give 24–30 Å gaps and materially
  larger PW/FFT cells for no physical gain. If 12 Å *per side* was truly intended,
  revert this amendment explicitly.
- **TMDs:** **rework faceting** — hex-aware Miller parsing + basal(0001) + S-deficient
  Mo-edge (the literature HER-active site), not just the 2H bulk fix.

## Workstreams (implement top-to-bottom, one branch)

### WS0 — Scaffolding (do first; enables safe testing)
- **CLI on `generate_structures.py`** (currently none): add `--include GLOB`
  (fnmatch over structure names), `--out-dir DIR` (default repo `DATA_INPUTS`; lets us
  regenerate a subset to a temp dir and diff before touching the repo), `--list`,
  `--dry-run`. Reuse the `fnmatch`/discovery convention already in
  `scripts/gpaw_h_adsorption.py:221 discover_structures`.
- **Validation helpers** (new small module or top of the file): `min_interatomic_distance(atoms)`
  (`get_all_distances(mic=True)`), `coordination(atoms, i, cutoff)`, and
  `assert_changed(child, parent)`.
- **Loud-failure plumbing:** `_write_structure` (L759) and the Part-1 inline block
  (L610) currently `try/except` and only *print* errors. Change so a builder that
  `raise`s → prints ✗, **removes any stale POSCAR at that path**, and records a failure
  that makes the run exit non-zero. This is what converts silent no-ops into visible
  failures.
- **Recreate `scripts/verify_inputs.py`** (the audit referenced in the plans but never
  committed): per POSCAR report min MIC distance, out-of-box fraction, and for every
  `*_vac*`/`*_dop*` assert it differs from its pristine parent (composition or coords).
  This is the acceptance gate, run after every regeneration.
  **[amended]** Use a **per-pair covalent-radius threshold** (flag any distance
  < 0.6·(rᵢ+rⱼ)), not a flat 0.7 Å: the "clean" `Ni_MoS2_interface_(111)` family has
  **0.92 Å** contacts (18 structures, re-verified 2026-07-09) — physically impossible
  for these elements, yet passing a 0.7 Å gate.

### WS1 — Base polymorphs (`create_*_bulk`, L15–96)
Build deterministically from documented Wyckoff/fractional coords, then assert
coordination before returning:
- **`create_mo2n_bulk`** → β anti-anatase I4₁/amd (mp-27953). 12-atom conv cell:
  Mo on the plain fcc lattice in an a×a×2a box; N on one octahedral site per (001)
  layer rotating 90° layer-to-layer (4₁ screw). **[amended]** The coords are NOT in
  `base_structure_polymorph_fix_plan.md` (that doc's γ rock-salt model is superseded);
  a verified construction (spglib → I4₁/amd #141, N=6 Mo, Mo=3 N): Mo at fcc sites
  {(0,0,0),(½,½,0),(½,0,¼),(0,½,¼)} + the same +(0,0,½); N at
  {(½,0,0),(½,½,¼),(0,½,½),(0,0,¾)}. Assert each N has 6 Mo and each Mo exactly 3 N
  with **±0.1 Å tolerance around 2.08 Å** — MP's relaxed mp-27953 distorts to Mo–N
  2.09/2.13 Å, so an exact-2.08 assertion would fail on relaxed cells.
- **`create_mop_bulk`** → WC-type P̄6m2 (mp-219), hex a≈3.23, c≈3.21: P (0,0,0),
  Mo (⅓,⅔,½). Assert both 6-coordinate, Mo–P ≈2.4–2.5 Å.
- **`create_mos2_bulk` / `create_mose2_bulk`** → 2H P6₃/mmc, **two** S–Mo–S layers
  (2 Mo + 4 X): a≈3.16 / 3.289, c≈12.295 / 12.929 (drop the 12.995), Mo 2c
  (⅓,⅔,¼ & ⅔,⅓,¾), X 4f. Assert 2 Mo + 4 X, each Mo 6-coord, interlayer gap present.
- **Out of scope but flag:** `create_mo2c_bulk` (L99) and `create_mob_bulk` (L125)
  also use hand-approximated coords; not in the three plans, leave as-is but let
  `verify_inputs.py` surface any coordination anomalies.

### WS2 — Hex-aware faceting + TMD basal/edge (`_parse_miller` L241, `create_slab` L249)
- Extend `_parse_miller` to accept 4-index hex (hkil) / a basal keyword, or add a
  parallel hex-slab path; route hexagonal/tetragonal bulks through it.
- **TMD (MoS₂/MoSe₂):** generate **basal (0001)** + **S-deficient Mo-edge** (extend the
  existing `create_edge_ribbon` edge path, removing chalcogen on the Mo-edge) instead of
  cubic (100)/(110)/(111). README notes basal is inert → edge/defect is the point.
- **MoP (hex):** basal (0001) + prismatic edge, same hex path.
- **Mo₂N (tetragonal β):** ordering breaks cubic degeneracy — generate the nonpolar
  **(001)ₜ + (100)ₜ** facets and the polar **Mo-terminated (111)/(112)** (cut in
  integer c/2 repeats; rely on the dipole correction from WS5). Compute both {100}
  variants — they bracket what disordered γ exposes.

### WS3 — Defect/dopant builders (L301–365) — vacancy_dopant plan A + B
- **A (anchor + raise):** in all three functions, replace whole-slab `z_max` with the
  **target species' own topmost z** (`z_top = max z over atoms of target species`);
  select within `tol` of `z_top`. If no candidate → **`raise`** (not `return slab`).
- **B1 (coverage):** build defect/dopant slabs on a **3×3** in-plane supercell (add a
  `defect_size` used by Parts 2/3, L620–683); log coverage. Keep the pristine reference
  at the same supercell (ΔG_H uses the defected slab as its own reference — already the
  case).
- **B2 (real clusters):** `create_multi_vacancy_slab` → seed site + its `count−1`
  **nearest same-species MIC neighbors**; assert contiguity with a per-material NN
  cutoff derived from the first-neighbor distance (not hard-coded).
- **B3:** drop the false "center isolates defect" comment; keep a deterministic pick.
- **B4 (structure side):** **[amended — dropped.]** `vacuum=8` (L249) is per side →
  the measured inter-image gap is already **16 Å** (≥12 Å standard met). Keep the
  geometry; the dipole correction (WS5) is the only real B4 fix. Note the defect-image
  distances motivating B1 are facet-dependent: 6.3 Å (MoS₂ slabs) but 8.3 Å
  (Mo₂N(100)) and 11.8 Å (Mo₂N(111)) — coverage argument stands, the "~6–7 Å" figure
  in the source plan only holds for the TMDs.

### WS4 — Interfaces: ZSL + dope-topmost-Mo (L402–541)
- Replace the concatenate-then-`set_cell(ni.cell)` logic in **`create_ni_mox_interface`**
  (L402) and **`create_ni_mxene_interface`** (L446) with pymatgen
  **`ZSLGenerator` / `CoherentInterfaceBuilder`** matching (strain tol <2%). Convert
  ASE↔pymatgen with `AseAtomsAdaptor` (pattern already in
  `scripts/compute_h_adsorption.py:162`). **Log induced strain + matched supercell size**
  per interface; if no match within tolerance → raise/skip (do not emit an overlap).
- **Dope topmost Mo:** `create_interface_with_dopant_generic` (L516) — change
  `target_symbol` default `"Ni"`→`"Mo"`; reuse the WS3 anchor+raise so an
  anion-terminated facet (no surface Mo) raises rather than doping a buried atom.
  **[amended]** Also fix or delete the dead non-generic duplicates
  `create_interface_with_dopant` (L530) and `create_interface_with_cluster` (L537) —
  unused by `generate_all_structures` but carrying the same `"Ni"` default.
- **[amended] Restore Ru to `decorations`** (L585, currently
  `["Ag","Au","Pd","Pt","Ir"]`): the headline `Ni_Mo2N_interface_(111)_cluster4Ru` has
  **no POSCAR in current inputs and cannot be generated by the current script** — its
  cluster-Ru inputs were dropped in an earlier regeneration; only stale UMA trajs /
  adsorbml results remain in `data/`. Without this, the regression test below is
  impossible.
- **Scope:** the builder becomes correct for all interface families, but re-screening
  is prioritized — rebuild/re-screen the headline `Ni_Mo2N_interface_(111)_cluster4Ru`
  and individually promising interfaces first; full 166-way interface screening is
  optional and expensive. **[amended]** Note the headline's −0.340 eV credential comes
  from the flagged-untrustworthy GPAW_LDA CSV — treat it as "candidate to re-derive",
  not established.

### WS5 — GPAW calculator (`scripts/gpaw_h_adsorption.py`) — B4 calculator side
- Enable the **dipole correction** in `GPAW_CONFIG` (top of file) —
  `poissonsolver={'dipolelayer': 'xy'}`; GPAW 25.7.0's `DipoleCorrection` explicitly
  supports the plane-wave implementation. **[amended]** Slabs stay fully periodic
  (PW mode requires 3-D PBC; the dipole layer handles the artificial field) — the
  original "z is non-periodic" wording was wrong for PW. Vacuum is already adequate
  (16 Å gap, see Decisions). This shifts **every** ΔG_H → full GPAW recompute
  (acceptable: existing ΔG_H CSVs are already flagged untrustworthy).
- **[amended] Decision point — OC20 consistency:** `GPAW_CONFIG` deliberately matches
  the OC20 reference *including* "no LDIPOL/IDIPOL", which is what UMA's `oc20` head
  was trained on. Enabling dipole in GPAW but not UMA introduces a systematic
  GPAW-vs-ML offset in the AdsorbML re-check comparison (`gibbs_free_ml_eV` vs GPAW
  ΔG_H). Physics-correct vs screening-consistent — pick consciously; if strict OC20
  match matters more, skip the dipole and flag polar-facet numbers instead.
- Ensure the **H₂ reference cache** (`data/outputs/h2_reference_energy.json`)
  invalidates on the changed config. **[amended]** The cache key is only
  steps/xc/mode/ecut (gpaw_h_adsorption.py:1089–1108) — a dipole key would NOT
  auto-invalidate it today, so this is a required code change, not a check. (H₂ itself
  is insensitive to a dipole layer; recomputing is cheap and keeps the key honest.)
- Note: UMA/AdsorbML relaxations don't use GPAW dipole, so this only re-runs the GPAW
  stage; UMA still re-runs because *structures* changed.

### WS6 — Downstream regeneration (one pass, tested first, user-triggered)
1. **Dry test:** `--include` a small subset with `--out-dir <tmp>` and
   `ADSORBML_DATA_ROOT=<scratch>`; run `verify_inputs.py`, then adsorbml 1→2→3 on a
   couple structures; sanity-check trajs.
2. **Full:** regenerate all (review `git diff` of `data/inputs/VASP_inputs/`), **delete
   stale cache** for changed names (`data/uma_relaxed/<name>.traj`+log,
   `data/adsorbml_results/<name>/`) since steps 1/2 skip on existing outputs, then
   `1-relax_uma_omat.py` → `2-run_adsorbml.py` → `3-extract_rank.py` → GPAW
   `--adsorbml-candidates`. Inputs are git-tracked (revertible); outputs are expensive
   (`Never clobber data/`) so this runs against real `data/` only after the temp test.

## Critical files
- `scripts/generate_structures.py` — WS0–WS4 (builders L15–96, `_parse_miller` L241,
  `create_slab` L249, defects L301–365, interfaces L402–541, `generate_all_structures`
  L549, `_write_structure` L759).
- `scripts/gpaw_h_adsorption.py` — WS5 `GPAW_CONFIG` + H₂ cache key; reuse
  `discover_structures` (L221) for the new `--include`.
- `scripts/verify_inputs.py` — **new**, WS0 audit gate.
- Reuse: `AseAtomsAdaptor` (`compute_h_adsorption.py:162`), `_apply_constraints` (L231),
  `create_edge_ribbon` (edge path), `_common.py` paths / `$ADSORBML_DATA_ROOT` (L28–36).

## Concerns
- **ZSL is the hardest new code.** `CoherentInterfaceBuilder` needs oriented slabs +
  termination selection and can fail to match within tolerance for some Ni/MoX pairs →
  must have a skip+log fallback. Exact pymatgen API depends on the installed version —
  verify against the env before relying on specifics.
- **β-Mo₂N is a surrogate**, not the real disordered/substoichiometric γ. Don't
  over-interpret ≲0.05 eV ordering differences; surface N-vacancy concentration may move
  ΔG_H more (the vacancy campaign partially covers this).
- **Dipole + Option B + supercell changes = essentially a full re-run** of UMA + GPAW
  across all affected families. This is a large compute commitment; it must be a
  deliberate, tested, user-triggered step, not an accidental clobber.
- **Polar Mo₂N (111)/(112) terminations** can be termination-dominated — a wrong
  termination is a bigger error than any ordering effect; cut in c/2 repeats + dipole.
- **MoP is also hexagonal**, so it needs the same hex faceting as the TMDs; don't leave
  it on cubic Miller cutting after WS2.
- **No `MP_API_KEY` is configured** — build Mo₂N/MoP from Wyckoff coords directly and
  cross-check against mp-27953/mp-219 numbers by hand; don't add a live MP dependency.
  (MP IDs verified 2026-07-09: mp-27953 = Mo₂N I4₁/amd #141; mp-219 = MoP P̄6m2 #187,
  Mo–P 2.46 Å.)
- **[amended] Dipole vs OC20/UMA consistency** (see WS5 decision point): the plan
  previously assumed "OC20-matched" and "dipole-corrected" simultaneously — they are
  mutually exclusive; one must be chosen.
- **[amended] Old cluster-Ru data are orphans.** `data/uma_relaxed/` and
  `data/adsorbml_results/` still hold `Ni_Mo2N_interface_*_cluster{2,4}Ru` outputs whose
  inputs no longer exist; any regeneration silently won't recreate them unless Ru is
  restored to `decorations` (WS4).

## Verification
- **Unit-level:** after WS1, print each base cell's composition + per-atom coordination
  and assert against the targets (Mo₂N N=6Mo@2.08±0.1 & Mo=3N; MoP 6-coord; TMD 2 Mo+4 X,
  Mo 6-coord). Fail the build on mismatch. Use a periodic-image-aware neighbor count
  (`ase.neighborlist`), not MIC pair distances — in small cells distinct octahedral
  neighbors collapse onto the same atom pair and undercount coordination.
- **Audit gate:** run `scripts/verify_inputs.py` over a temp regeneration — expect
  **0 structures with any pair below the covalent-radius threshold** (not just <0.7 Å;
  see WS0) and **0 `*_vac*`/`*_dop*` identical to parent**. This directly refutes bugs
  #2 and #3.
- **End-to-end smoke:** on a temp `--out-dir` + `$ADSORBML_DATA_ROOT`, run one
  Mo₂N slab and one rebuilt interface through `1→2→3` then a single GPAW point; confirm
  it completes and ΔG_H is finite/sane.
- **Regression on headline:** rebuild `Ni_Mo2N_interface_(111)_cluster4Ru` with ZSL —
  **requires Ru restored to `decorations` first (WS4)**, since the current generator
  cannot produce it — confirm no sub-covalent-radius contacts and logged strain <2%,
  and that it now stands on the correct β-Mo₂N base.
