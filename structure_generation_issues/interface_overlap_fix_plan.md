# Fix plan — interface builder atom overlaps (lattice mismatch)

**Status:** not implemented (plan only). Written 2026-07-08.
**Scope:** the two interface builders in `scripts/generate_structures.py`
(`create_ni_mox_interface` ~L402 and `create_ni_mxene_interface` ~L446). This is
issue #3 of the input-generation defects. Distinct from — but shares the same
builder as — the interface dopant no-op in
`vacancy_dopant_and_methodology_fix_plan.md` (issue #2), so **land both in one
interface-builder pass**. Also distinct from the wrong base polymorphs
(`base_structure_polymorph_fix_plan.md`, issue #1).

---

## What is wrong (verified 2026-07-08: audit + code)

**54 `Ni_*_interface_*` structures are physically invalid** — atoms wrap to
**0.22–0.43 Å apart** under periodic boundary conditions. No two atoms in any real
material sit closer than ~0.7 Å even in the tightest bond, so 0.2 Å means two
distinct atoms have collapsed onto essentially the same point.

Concrete damage (from the 2026-07-08 input audit, `get_all_distances(mic=True)` +
out-of-box scaled-position fraction on orthogonal cells, so not a shear/MIC
artifact):
- `Ni_MoB_interface_(100)_*` — min dist **0.22 Å**, ~100/164 atoms outside the
  in-plane cell.
- `Ni_MoB_interface_(111)_*` — min dist **0.32 Å**.
- `Ni_Mo2C_interface_(111)_*` — min dist **0.43 Å**, ~41/84 atoms outside the cell.

Affected families are ~18 variants each (pristine + dopants + clusters).

**Root cause (in code):** the builder stacks the substrate slab and the Ni block
on **inconsistent in-plane lattices** — e.g. Ni `(3,3,4)` and MoX `(2,2,2)` with
different in-plane cell sizes — then concatenates (`ni + mox`) and stamps the Ni
cell onto the result (`set_cell(ni.cell)`). It **never lattice-matches** the two
pieces. The final cell is only ~7.47 × 7.47 Å while the atoms actually span ~30 Å
in-plane, so ~half the atoms fall outside the box and periodic wrapping smashes
distinct atoms onto the same site. `create_ni_mxene_interface` (~L446) has the
identical flaw.

**Why it's a mistake:** an energy on atoms 0.2 Å apart is meaningless (or SCF
fails to converge). These are not the material their filename claims → any result
computed on the 54 is void.

**What is NOT confirmed broken:** other interfaces (`Ni_Mo2N`, `Ni_MoS2`,
`Ni_Ti3C2O2`) have un-wrapped coordinates and short contacts but **no <0.7 Å
overlap** — suspicious, not definitively broken. *(Corrected 2026-07-09: measured
per-family minima are 0.92 Å for `Ni_MoS2_(111)` — below the "~1.1–1.3 Å" originally
reported and physically impossible for these elements — then 1.07–1.37 Å for the
rest. The 18 `Ni_MoS2_interface_(111)_*` structures should be treated as broken too;
a flat 0.7 Å audit threshold is too lenient — use a per-pair covalent-radius
threshold.)* The headline HER candidate `Ni_Mo2N_interface_(111)_cluster4Ru` is
**not** in the overlap set — but see the 2026-07-09 caveat below: it no longer exists
in the current inputs at all.

**Contrast with issue #2:** this bug is **not silent** — overlaps produce
obviously broken energies or crashes, so it announces itself downstream rather
than masquerading as valid (unlike the silent no-op vacancy/dopant).

---

## How it could be fixed

The overlaps come from gluing two crystals with **different in-plane
periodicities** into one periodic box. The fix is to make the two grids
compatible **before** combining — i.e. lattice-match — and centralize
stack+match in **one helper** shared by both interface builders (also the place
to fix the issue-#2 "dope topmost Mo" target).

### Option A — pragmatic: strain the overlayer onto the substrate
Force MoX to adopt Ni's in-plane cell, moving atoms with the cell so nothing
spills outside:
```
mox.set_cell([ni.cell[0], ni.cell[1], mox.cell[2]], scale_atoms=True)
# then concatenate
```
- **Pros:** one line, cheap, keeps atom counts small.
- **Cons:** if the Ni and MoX lattices differ much, this imposes **>5–10%
  in-plane strain**. Geometry becomes valid (no overlaps) but the *energies* on
  these interfaces stay physically suspect. Acceptable for coarse trend-level
  UMA screening only, **if the induced strain is logged and thresholded**.

### Option B — rigorous: coincidence-site lattice matching
Use `pymatgen` `ZSLGenerator` / `CoherentInterfaceBuilder` (pymatgen is already a
dependency) to find a supercell where both lattices line up within a chosen
strain tolerance (e.g. <2%).
- **Pros:** low, controlled strain → energies you can defend.
- **Cons:** more atoms (bigger supercell), more compute; **changes atom counts**.

### What does NOT work
**Wrapping atoms back into the box is not a fix** — it relocates the collision,
it does not remove it. The two atoms are still coincident; only which periodic
image you look at changes. The problem is the mismatched cells, not the wrapping.

---

## Is it even worth screening the interface family?

Recommendation: **broad interface screening is low priority; do targeted
follow-up instead.**
- Interfaces are the most broken (issues #1+#2+#3 all concentrate here) and the
  most expensive to model correctly.
- The project's own direction (README) is edges + defects on **Mo₂N**, which is
  cheaper (smaller cells) and where the literature signal is — not Ni interfaces
  broadly.
- But the current headline candidate *is* an interface
  (`Ni_Mo2N_interface_(111)_cluster4Ru`), so the move is **targeted, not mass**:
  rigorously rebuild + validate that one (and any individually promising
  interface) with Option B straight through GPAW, rather than fixing and
  re-screening all ~54 variants.
- **Caveat (verified 2026-07-09):** that headline structure has **no POSCAR in the
  current inputs** and the current generator **cannot produce it** — `decorations`
  (generate_structures.py L585) is `["Ag","Au","Pd","Pt","Ir"]`, no Ru; only stale
  UMA trajs / adsorbml outputs remain in `data/`. Rebuilding it requires restoring
  Ru to the decorations list first. Its −0.340 eV ΔG_H also comes from the
  flagged-untrustworthy GPAW_LDA CSV — treat "headline" as "to re-derive", not
  established.

---

## Downstream consequences (after any interface-builder change)

- Both options invalidate the interface families: **regenerate → re-relax (UMA) →
  re-rank**. Option B changes atom counts, so a full re-run of the interface
  families is required regardless.
- **Sequencing** (do not waste regeneration cycles):
  1. Rebuild wrong base polymorphs (`base_structure_polymorph_fix_plan.md`).
  2. Fix this overlap bug **and** the issue-#2 "dope topmost Mo" target in one
     interface-builder pass (`vacancy_dopant_and_methodology_fix_plan.md` A#3).
  3. One regeneration + re-relax + re-rank over the affected families.
- **Never clobber `data/`** — test regeneration against a temp
  `$ADSORBML_DATA_ROOT` or a copied subset first, then diff before replacing.

---

## Concerns

- **The headline candidate IS standing on the wrong Mo₂N base** (verified
  2026-07-09). The audit called `Ni_Mo2N_interface_(111)_cluster4Ru` "clean," but
  only for the overlap (#3) and no-op (#2) bugs. `create_ni_mox_interface` calls
  `mox_bulk_builder()` directly (generate_structures.py L411), i.e. it slices its
  Mo₂N from the buggy `create_mo2n_bulk` (wrong body-centered basis, issue #1) —
  so the clean-celled headline result rests on a wrong crystal and does not
  survive the base fix.
- **Option A (strain) can silently degrade energy quality.** Strained structures
  look fine (no overlaps) and relax cleanly, so this can re-hide the way issue #2
  did — unless the induced strain is explicitly logged and thresholded per
  interface.
- **Sequencing is not optional.** "Dope topmost Mo" and "correct facet
  termination" depend on the interface geometry being right first. Fixing the
  dopant target before the overlap/lattice-match risks identifying "topmost Mo"
  on a broken surface.
- **"Not worth screening" ≠ "worthless."** Ni-support effects on HER are real
  physics; the problem is the current inputs, not the idea. If interfaces are
  central to the paper's story, use Option B — it's just expensive.
- **Verify against current code before implementing** — builders are hand-typed
  and may have been edited since this note; re-read `create_ni_mox_interface`
  (~L402) and `create_ni_mxene_interface` (~L446) first.

---

## Sources / references

- Overlap audit: agent-memory `interface-builder-overlap-bug` (54 structures,
  0.22–0.43 Å min distances, out-of-box fractions), `input-generator-fix-plan`
  (root cause: Ni `(3,3,4)` + MoX `(2,2,2)` concatenated then `set_cell(ni.cell)`,
  no lattice match; `create_ni_mxene_interface` same flaw). Audit script:
  scratchpad `verify_inputs.py` (2026-07-08 session).
- Interface geometry (MoX stacked on top of buried Ni; dopant target currently
  `"Ni"`): code `generate_structures.py` L402–L437, L516; see
  `vacancy_dopant_and_methodology_fix_plan.md`.
- Lattice matching: `pymatgen` `ZSLGenerator` / `CoherentInterfaceBuilder`
  (already a project dependency).

Related: `base_structure_polymorph_fix_plan.md` (issue #1),
`vacancy_dopant_and_methodology_fix_plan.md` (issue #2), agent-memory
`interface-builder-overlap-bug`, `dft-reference-data-broken`.
