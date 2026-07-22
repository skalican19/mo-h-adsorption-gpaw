# Fix plan — vacancy/dopant generation: silent no-op bug + methodology

**Status:** not implemented (plan only). Written 2026-07-08.
**Scope:** the defect/doping builders in `scripts/generate_structures.py`
(`create_vacancy_slab` L301, `create_multi_vacancy_slab` L323,
`create_substitution_slab` L347, and the interface wrappers L516/L523).
Covers **(A)** the silent no-op code bug and **(B)** the methodology weaknesses
in how vacancies/dopants are built. All line numbers verified against current
code 2026-07-08.

**Out of scope (separate plans):** wrong base polymorphs
(`base_structure_polymorph_fix_plan.md`) and the interface atom-overlap bug
(agent-memory `interface-builder-overlap-bug`). This plan touches the same
interface builder as the overlap bug, so **sequence them together** — see
Downstream.

---

## Part A — silent no-op vacancy/dopant (the code bug)

### Root cause (verified)
All three functions do the same thing:
```
z_max = np.max(z_positions)                                   # highest atom of ANY species
candidates = [i for ... if symbol == target and (z_max - z[i]) < 0.5]
if not candidates: return slab                                # silent no-op, no warning
```
The window is anchored to the **whole-slab** `z_max` with a fixed 0.5 Å
tolerance. If no target-species atom sits within 0.5 Å of the global ceiling,
the candidate list is empty and the function **returns the slab unchanged with
no error/log**. The output file is written under the defect's name but is a
byte-identical copy of the pristine parent.

### Impact (from the 2026-07-08 input audit; re-verified 2026-07-09)
**63 structures are byte-identical to their pristine parent because of this bug**
(strict byte-identical count is **71**: the extra 8 are interface `dopNi`, identical
by definition since they substitute Ni→Ni — after the A#3 retarget to Mo they become
meaningful dopants):
- **8 vacancies** — target atom sits below the top layer on certain facets
  (e.g. Mo₂N(111) terminates in pure Mo; N is ~1.2 Å down, outside the 0.5 Å
  window).
- **55 dopants** — dominated by interfaces: `create_ni_mox_interface`
  (L402) stacks **MoX on top of Ni** (Ni buried), but
  `create_interface_with_dopant_generic` (L516) calls
  `create_substitution_slab(interface, target_symbol="Ni", dopant)`. Ni is at
  the bottom, so no Ni is near the top → every interface substitutional dopant
  is a no-op (and interface `dopNi` too). Plus the Mo₂C(111) dopants.

### Fix
1. **Anchor the window to the target species' own topmost z**, not the global
   `z_max`:
   `z_top = max(z[i] for i where symbol == target)`; select within a tolerance
   of `z_top`. Drops the "defect must be the highest atom in the cell"
   assumption.
2. **Never return silently.** If no candidate qualifies, `raise` (or at minimum
   `log` loudly and mark the structure skipped) so a failed edit can't
   masquerade as a real structure. This is the single most important change —
   it converts every future occurrence from a silent duplicate into a visible
   failure.
3. **Interface dopants — target the exposed cation, not Ni** (confirmed
   decision, user 2026-07-08). The exposed catalytic surface is MoX, so dope
   the **topmost Mo** site. Change `create_interface_with_dopant_generic`'s
   `target_symbol` default from `"Ni"` to `"Mo"` (or pass it explicitly per
   family). **If the exposed facet is anion-terminated** (no surface Mo; topmost
   Mo is more than ~1 layer down) → **raise/skip**; do not place a buried
   subsurface dopant (unphysical, no effect on surface H). Also fix or delete the
   dead non-generic duplicates `create_interface_with_dopant` (L530) and
   `create_interface_with_cluster` (L537) — unused, same `"Ni"` default.
   - Interpretation: these model a noble-metal-doped MoX surface *supported on*
     Ni — the Ni is an electronic/strain substrate, you are NOT doping the
     Ni/MoX contact.

### Verify after fix
- Assert every emitted `*_vac*`/`*_dop*` file differs from its pristine parent
  (composition or coordinates). A cheap post-generation check: hash each POSCAR
  against its parent and fail the build on a collision.
- Re-run the audit script (scratchpad `verify_inputs.py`, 2026-07-08 session).

---

## Part B — methodology fixes

The concept (surface vacancy = under-coordinated site; single-atom
substitutional dopant on the top surface) is standard and correct for HER
screening. The mechanics as built are not quantitative:

### B1. Defect concentration far too high — **biggest issue**
`create_slab` uses `size=(2,2,4)` (L249). One defect per 2×2 in-plane cell puts
the defect and its periodic images close together, interacting strongly. These
are **defect-superlattice** energies, not isolated-defect energies.
*(Corrected 2026-07-09: the image distance is facet-dependent — measured 6.3 Å for
MoS₂ slabs, 8.3 Å for Mo₂N(100), 11.8 Å for Mo₂N(111). The coverage argument stands;
"~6–7 Å" only holds for the TMDs.)*
- **Fix:** build defect/dopant slabs on a larger in-plane supercell (**3×3 or
  4×4**) so coverage drops to ~1/9–1/16 and image interaction is small. Keep the
  pristine reference at the **same** supercell (ΔG_H uses the defected slab as
  its own reference).
- **Tradeoff:** cost scales with atom count. 4×4 is cheap for UMA screening;
  for the GPAW re-check, 3×3 may be the practical compromise. Whatever is
  chosen, **log the coverage** so the number's meaning is explicit.

### B2. "Vacancy cluster" is not guaranteed to be a cluster
`create_multi_vacancy_slab` (L323) removes the `count` atoms **closest to the xy
center**, not atoms adjacent to each other. A di-/tri-vacancy is meant to be
neighboring sites; nearest-to-center can pick separated vacancies mislabeled as
a cluster.
- **Fix:** pick a seed site, then add its `count-1` **nearest same-species
  neighbors** (minimum-image distance) to form a bonded cluster. Optionally
  assert the removed set is mutually contiguous (max pairwise MIC distance below
  a nearest-neighbor cutoff).

### B3. "Closest to xy center" is a no-op for periodic slabs
Under PBC every equivalent top-layer site of a species is identical to its
images regardless of xy position, so centering neither isolates the defect nor
reduces image interaction. It is harmless but the implied rationale is false
(it only matters for the non-periodic ribbon/edge cases).
- **Fix:** keep a deterministic pick for reproducibility, but drop/adjust the
  "center isolates the defect" reasoning in comments. Real isolation comes from
  B1 (bigger cell), not from site choice.

### B4. Asymmetric slab + no dipole correction
`create_slab` freezes the bottom half, so a defect/dopant on one face makes the
slab polar; with no dipole correction a spurious field crosses the cell and
biases ΔG_H. Affects all these slabs but is worst for polar terminations (e.g.
any pure-Mo Mo₂N facet).
- **Corrected 2026-07-09 — the "thin vacuum" half of this item was wrong:**
  `vacuum=8` in ASE is *per side*; the measured inter-image gap on the actual
  POSCARs is **16 Å**, already above the ≥12 Å standard. Raising the parameter
  to 12–15 would give 24–30 Å gaps and larger PW/FFT cells for no gain.
- **Fix (one file):** enable a **dipole correction** in the GPAW calculator
  config in `scripts/gpaw_h_adsorption.py` (calculator side only; keep the slab
  geometry). Cross-cutting — applies beyond vacancies/dopants. See the
  implementation plan's WS5 for the OC20-consistency decision this forces.

---

## Downstream consequences (must-do after these fixes)

- Any structure whose builder changes is invalidated: **regenerate → re-relax
  (UMA) → re-rank** the affected families. Existing `data/uma_relaxed/` trajs
  and ranked candidates for the 63 no-op structures (and all defect/dopant
  slabs, once the supercell changes) become stale.
- **Sequencing** (do not waste regeneration cycles):
  1. Rebuild wrong base polymorphs (`base_structure_polymorph_fix_plan.md`).
  2. Fix the interface **atom-overlap** bug (agent-memory
     `interface-builder-overlap-bug`) — same builder as A#3 here, so land the
     overlap fix and the "dope topmost Mo" fix in one interface-builder pass.
  3. Fix this plan (A + B).
  4. **One** regeneration + re-relax + re-rank over all affected families.
- **Never clobber `data/`.** Test regeneration against a temp
  `$ADSORBML_DATA_ROOT` or a copied subset first, then diff before replacing.

---

## Concerns

- **B1 supercell size changes atom counts** for every defect/dopant slab, so it
  invalidates far more cached data than the code bug alone — not just the 63
  no-ops. Budget for a full re-run of all defect/dopant families, and pick the
  supercell once (changing it later forces another re-run).
- **B4 dipole correction is a calculator change**, not a structure change, and
  will shift *every* ΔG_H (including already-computed pristine slabs), so it
  breaks comparability with any earlier numbers unless everything is recomputed.
  Decide whether to adopt it now (clean but a full recompute) or defer it and
  flag current polar-facet numbers as biased.
- **Interface "dope topmost Mo" depends on a correct facet termination.** If the
  base-polymorph and overlap bugs are not fixed first, "topmost Mo" may be
  identified on a wrong or overlapped surface — hence the sequencing above.
  Doing A#3 before steps 1–2 risks doping the wrong site.
- **B2 contiguity assumes a well-defined nearest-neighbor cutoff**, which varies
  by material; a fixed cutoff may misbehave for the softer/longer bonds. Derive
  the cutoff per material from the first-neighbor distance rather than hard-coding.
- **Verify against current code before implementing** — builders are hand-typed
  and may be edited independently of this note; re-read the five functions
  first.

---

## Sources / references

- Silent no-op audit + interface geometry: agent-memory
  `vacancy-111-generation-bug`, `input-generator-fix-plan` (confirmed
  "dope topmost Mo, else raise/skip" decision, user 2026-07-08). Code verified:
  `generate_structures.py` L301/L323/L347 (defect fns), L402–L437
  (`create_ni_mox_interface`, MoX-on-Ni stacking), L516 (dopant target="Ni").
- ASE constraint behavior: verified 2026-07-08 that `del atoms[i]` re-maps
  `FixAtoms` indices correctly, so deleting a top-layer atom does **not** corrupt
  the frozen-bottom constraint (no fix needed there).
- Defect-supercell / dipole-correction rationale is standard DFT-slab practice
  (isolate defects by dilution; correct asymmetric-slab dipole; ≥12 Å vacuum
  with an adsorbate).

Related: `base_structure_polymorph_fix_plan.md`, agent-memory
`interface-builder-overlap-bug`, `dft-reference-data-broken`.
