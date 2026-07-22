# Fix plan — wrong base crystal polymorphs in `generate_structures.py`

**Status:** not implemented (plan only). Written 2026-07-08.
**Scope:** the three base bulk builders that produce the *wrong polymorph/basis*.
This is a **separate, deeper defect class** than the two known builder bugs
(silent no-op vacancy/dopant; interface atom overlaps) — those are tracked in the
agent-memory `input-generator-fix-plan`. Rebuilding the bases does **not** fix
those two, and fixing those two does **not** fix this. Order matters: **fix the
bases first**, because every slab/defect/edge/interface is sliced from them.

Why relaxation does not help: geometry relaxation is a *local* optimizer — it
settles atoms into the nearest energy minimum. A wrong polymorph is a different
structure (different bonding topology / different energy basin), so relaxation
cannot convert it to the right one; it will "converge" on the wrong structure and
mask the error. For the TMD monolayer-as-bulk case relaxation cannot add the
missing second layer at all (atom count is fixed). The base cell must be rebuilt
from a correct reference.

---

## What is wrong (verified 2026-07-08 against code + external sources)

Code confirmed by reading `scripts/generate_structures.py`
(`create_mop_bulk`, `create_mo2n_bulk`, `create_mos2_bulk`, `create_mose2_bulk`).

### 1. MoP — wrong polymorph
- **Code builds:** cubic CsCl / B2 (`a=3.240`; Mo at (0,0,0), P at (½,½,½)).
- **Correct:** hexagonal **WC-type, space group P̄6m2 (No. 187)**, a≈3.23 Å,
  c≈3.21 Å. Mo and P are both 6-coordinate (trigonal-prismatic). Wyckoff:
  P at 1a (0,0,0), Mo at 1d (⅓,⅔,½). (Materials Project **mp-219**.)
- **Consequence:** every MoP slab is a nonexistent phase.

### 2. Mo₂N — wrong basis (headline material)
- **Code builds:** cubic `a=4.160`; Mo at (0,0,0)+(½,½,½) [body-centered, *not* fcc]
  and N at (½,½,0). Code comment mislabels it "anti-perovskite."
- **Correct:** γ-Mo₂N = **fcc rock-salt, space group Fm‑3m (No. 225)**, a≈4.16 Å:
  an fcc array of Mo with N occupying **half the octahedral interstitial sites**
  (each N ideally 6-coordinate octahedral at ~2.08 Å). The lattice constant is
  right; the atom arrangement is wrong (buggy cell gives N 2-fold at 2.08 Å +
  4-fold at 2.94 Å instead of 6-fold octahedral).
- **N-site disorder caveat:** in real γ-Mo₂N the half-occupancy is *random*
  (disordered). A DFT/ML model needs an **ordered approximation** of "half the
  octahedral sites filled" — this is a modeling decision (see Concerns), not a
  single canonical cell.
- **Consequence:** Mo₂N is the top HER candidate; this error lands on the most
  important result.

### 3. MoS₂ / MoSe₂ — monolayer mislabeled as bulk
- **Code builds:** hexagonal cell with the correct c (MoS₂ c=12.295, MoSe₂
  c=12.995) but only **one** S–Mo–S layer in it (1 Mo + 2 S).
- **Correct:** 2H polytype, **space group P6₃/mmc (No. 194)** — **two** S–Mo–S
  layers per c-cell in ABA stacking (2 Mo + 4 S), a≈3.16 Å (MoS₂) / 3.289 Å
  (MoSe₂), c≈12.295 Å. Mo at 2c, S at 4f.
- **Consequence:** it is an isolated monolayer sitting in a box sized for two
  layers; slicing (100)/(110)/(111) off it yields ill-defined terminations.

---

## Fix approach

Preferred: build each cell **deterministically from documented Wyckoff positions**
(no live network/API dependency), then cross-check the result against the
Materials Project entry. `pymatgen` is already a dependency and can pull the
reference `Structure`/CIF for verification (or as the source itself if an API key
is configured).

### MoP → WC-type (P̄6m2, 187)
- Hexagonal cell: a=b≈3.23, c≈3.21, γ=120°.
- Basis (fractional): P (0,0,0); Mo (⅓,⅔,½).
- Sanity check after build: both atoms 6-coordinate; nearest Mo–P ≈ 2.4–2.5 Å.

### Mo₂N → rock-salt (Fm‑3m, 225) with an ordered half-N model
**[SUPERSEDED 2026-07-09 — see `implementation_plan.md` "Decisions locked" / WS1.]**
The locked decision is **β-Mo₂N anti-anatase, I4₁/amd (No. 141), matching mp-27953**
(the ordered ground state; verified: MP confirms I4₁/amd, and the WS1 construction
reproduces it with every N = 6 Mo and every Mo = exactly 3 N). The text below is the
original rock-salt-derived proposal, kept for history:
- Start from ideal rock-salt MoN (Mo fcc + N in all octahedral holes), then
  remove **half** the N in an *ordered* pattern to reach Mo₂N stoichiometry.
- Recommended ordered model: use the smallest supercell that gives an even,
  charge-reasonable N arrangement (e.g. alternating filled/empty octahedral
  sites). **Decide and document the ordering** — do not leave it random.
- Sanity check: every N octahedrally coordinated by 6 Mo at ~2.08 Å; overall
  Mo:N = 2:1.

### MoS₂ / MoSe₂ → 2H (P6₃/mmc, 194), two layers
- Hexagonal cell: a≈3.16 (S) / 3.289 (Se), c≈12.295 (S) / 12.929 (Se — note the
  code's 12.995 is ~0.5–0.7% high; take the literature value).
- Basis: 2 Mo (2c: ⅓,⅔,¼ and ⅔,⅓,¾) + 4 chalcogen (4f), reproducing ABA stacking
  with trigonal-prismatic Mo.
- Sanity check: 2 Mo + 4 X per cell; each Mo 6-coordinate; interlayer gap present.

---

## Downstream consequences (must-do after any base rebuild)

- Every structure derived from a fixed base is invalidated: **regenerate →
  re-relax (UMA) → re-rank** all MoP, Mo₂N, and MoS₂/MoSe₂ slabs, defects, edges,
  and interfaces. Existing `data/uma_relaxed/` trajs and ranked candidates for
  those families become stale.
- Sequence the whole repair as: **(a)** rebuild bases (this plan) → **(b)** fix the
  vacancy/dopant no-op + interface-overlap builder bugs (`input-generator-fix-plan`)
  → **(c)** one regeneration + re-relax + re-rank pass over the affected families.
  Doing (b) before (a) wastes a regeneration cycle.
- `Never clobber data/` still applies — test regeneration against a temp
  `$ADSORBML_DATA_ROOT` or a copied subset first.

---

## Concerns

- **Mo₂N ordered-cell choice is a real scientific decision, not a mechanical fix.**
  Different ordered arrangements of the half-filled N sublattice give different
  surfaces and different ΔG_H. Pick a defensible ordering and document it; ideally
  check that a couple of orderings give consistent trends before trusting the
  ranking. This is the highest-judgment part of the plan.
- **Cubic Miller indices on hexagonal materials (MoP, MoS₂, MoSe₂) is still wrong
  even after the base is fixed.** `_parse_miller` only accepts 3 single digits, so
  hex cells are cut with cubic indices; the physically relevant TMD **basal (0001)
  and edge** surfaces are never generated. For MoS₂ the literature HER-active site
  is the **S-deficient Mo-edge**, which the current edge builder does not
  reproduce. Fixing the bulk alone will still leave the *faceting strategy* wrong
  for the layered materials — decide separately whether to switch to
  basal+edge generation for TMDs.
- **Vacuum / dipole:** asymmetric slabs with only 8 Å vacuum and no dipole
  correction bias polar facets (e.g. any pure-Mo–terminated Mo₂N facet). ≥12 Å
  vacuum is standard once an adsorbate is present. Orthogonal to the polymorph fix
  but affects every ΔG_H.
- **Verify against current code before implementing.** Builders are hand-typed and
  may have been edited since this note; re-read the four functions first.
- This plan changes atom counts and cell contents for ~all MoP/Mo₂N/TMD inputs, so
  it is not reversible against existing cached results — budget for a full re-run
  of those families.

---

## Sources

- MoP — hexagonal WC-type, P̄6m2 (187): Materials Project **mp-219**
  <https://next-gen.materialsproject.org/materials/mp-219>
- γ-Mo₂N — cubic rock-salt Fm‑3m, N on half the octahedral sites:
  - *Molybdenum Nitride Films: Crystal Structures, Synthesis, …*, Coatings 5(4):656, MDPI
    <https://www.mdpi.com/2079-6412/5/4/656>
  - *Cation and anion vacancies in cubic molybdenum nitride*, J. Alloys Compd.
    <https://www.sciencedirect.com/science/article/abs/pii/S0925838817304887>
- MoS₂ — 2H polytype, P6₃/mmc (194), two S–Mo–S layers, c≈12.295 Å:
  - SpringerMaterials (Sharma 2011): <https://materials.springer.com/isp/crystallographic/docs/sd_1826317>
  - Materials Project **mp-1018809**: <https://next-gen.materialsproject.org/materials/mp-1018809>
- MoSe₂ — 2H, P6₃/mmc (Hotje 2005):
  <https://materials.springer.com/isp/crystallographic/docs/sd_1937551>

Related agent-memory notes: `base-structure-polymorph-errors` (findings),
`input-generator-fix-plan` (the other two builder bugs), `dft-reference-data-broken`.
