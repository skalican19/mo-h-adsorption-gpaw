"""
scripts/adsorbml/tagging.py

Surface tagging for the AdsorbML pipeline: decide which slab atoms are "surface"
(tag 1, free to relax) and which stand in for bulk (tag 0, frozen). `tag=2` is
reserved for adsorbates and is never produced here.

Why this is its own module
--------------------------
Tags decide THREE things in fairchem, not one:

  1. which atoms relax          FixAtoms(mask=tag==0), steps 1 and 2
  2. where H is placed          Delaunay mesh over tag-1 atoms only
                                (fairchem/data/oc/core/adsorbate_slab_config.py)
  3. what counts as an anomaly  is_adsorbate_intercalated rejects any placement
                                whose H neighbours ANY tag-0 atom
                                (fairchem/data/oc/utils/flag_anomaly.py)

Because of (3), a tag-0 atom that H can physically touch turns every placement over
it into a false "intercalated" rejection. That is what silently produced ZERO
candidates for all 48 Mo2N structures: gamma-Mo2N(001) has ordered N-vacancy pits
whose floor is a second-layer Mo exactly 2.00 A down -- on the wrong side of the old
`z > z_max - 2.0` rule in 1-relax_uma_omat.py. A static audit found 219/350 inputs
with at least one frozen-but-reachable atom; the cluster-on-slab interfaces were
worse than Mo2N, freeing as few as 5 atoms of 262 because the cluster apex set z_max.

Upstream's own tagger (fairchem.data.oc.core.slab.tag_surface_atoms) does NOT fix
this. Measured on Mo2N_(001), it frees the same 12/192 atoms as the old rule:

  - its height test compares scaled coordinates with a strict `<`, which for an
    interlayer spacing equal to the window is decided by ~1e-16 of floating-point
    noise (threshold 0.782608695652174 vs layer-2 0.78260869565217384);
  - its Voronoi test promotes an atom only when its CN falls below min(bulk CN), and
    a substoichiometric nitride's bulk already contains low-coordination metal, so
    the pit floor is not anomalous by that measure -- the caveat in fairchem's own
    docstring.

So OC20's height criterion is kept as the compatible baseline (that is the convention
UMA's `oc20` head and the anomaly filter both assume), but the load-bearing criterion is
the reachability invariant, with a layer floor underneath it. OC20's under-coordination
criterion was implemented and removed — see `surface_tags` for the measurement.

Run `python scripts/adsorbml/audit_tags.py` after any change to this rule or to the
structure generator: it re-checks the invariant over every input and exits non-zero on
failure, so it can gate a submission script. Currently 0/350 fail.
"""
import numpy as np

# --- Constants ---------------------------------------------------------------
TAG_HEIGHT_WINDOW = 2.0    # Å, OC20's "within 2 Å of the topmost atom"
TAG_MIN_LAYERS    = 2      # minimum atomic layers freed (the standard slab convention)
TAG_LAYER_TOL     = 0.4    # Å, z spread within which atoms count as one layer
TAG_THIN_LAYERS   = 3      # a stack this shallow has no interior -> free all (see below)
TAG_PROBE_GRID    = 0.35   # Å, raster spacing of the reachability test
TAG_PROBE_RADIUS  = 0.31   # Å, covalent radius of H — the probe we must not freeze under
TAG_OVERFREE_WARN = 0.7    # warn if more than this fraction of a thick slab is freed


# --- Building blocks ----------------------------------------------------------
def layer_groups(z, tol=TAG_LAYER_TOL) -> list:
    """Group atom indices into atomic layers by z, topmost layer first.

    Membership is measured against the layer's highest atom rather than the previously
    added one, so a gently sloping surface cannot chain one "layer" down the whole slab.
    """
    z = np.asarray(z)
    order = list(np.argsort(-z))
    groups, current = [], [order[0]]
    for i in order[1:]:
        if z[current[0]] - z[i] <= tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def exposed_from_above(atoms, probe_radius=TAG_PROBE_RADIUS, grid=TAG_PROBE_GRID) -> set:
    """Indices of atoms a probe of `probe_radius` can touch coming from +z.

    One-sided on purpose. AdsorbML only ever places the adsorbate above the slab (the
    Delaunay mesh is built in xy and H is dropped from above), so the bottom face is
    unreachable by H and must stay frozen — it is the artificial cut standing in for
    bulk material below.

    Method: rasterise the cell in fractional xy, and at each grid point keep whichever
    atom's inflated sphere (covalent radius + probe radius) reaches highest. Any atom
    winning at least one grid point is reachable. This is a discretised
    solvent-accessible-surface test; `grid` bounds the narrowest pit it can resolve.
    """
    from ase.data import atomic_numbers, covalent_radii

    pos = atoms.positions
    radii = np.array([covalent_radii[atomic_numbers[s]]
                      for s in atoms.get_chemical_symbols()])
    cell = atoms.cell.array

    # Nothing far below the top can win a grid point; 8 Å is a generous, cheap bound
    # that keeps the raster cost independent of slab thickness.
    candidates = np.where(pos[:, 2] > pos[:, 2].max() - 8.0)[0]
    if candidates.size == 0:
        return set()

    a_len, b_len = np.linalg.norm(cell[0]), np.linalg.norm(cell[1])
    na, nb = max(2, int(np.ceil(a_len / grid))), max(2, int(np.ceil(b_len / grid)))
    fa, fb = np.meshgrid(np.arange(na) / na, np.arange(nb) / nb, indexing="ij")
    points = fa.ravel()[:, None] * cell[0][:2] + fb.ravel()[:, None] * cell[1][:2]

    # Neighbouring cell images: a pit near the cell boundary is fed by atoms across it.
    shifts = [i * cell[0][:2] + j * cell[1][:2] for i in (-1, 0, 1) for j in (-1, 0, 1)]

    best = np.full(len(points), -np.inf)
    winner = np.full(len(points), -1, dtype=int)
    for i in candidates:
        reach = radii[i] + probe_radius
        for shift in shifts:
            d2 = ((points - (pos[i, :2] + shift)) ** 2).sum(axis=1)
            hit = d2 < reach * reach
            if not hit.any():
                continue
            height = pos[i, 2] + np.sqrt(reach * reach - d2[hit])
            idx = np.where(hit)[0]
            better = height > best[idx]
            best[idx[better]] = height[better]
            winner[idx[better]] = i
    return set(int(i) for i in winner[winner >= 0])


# --- The rule -----------------------------------------------------------------
def surface_tags(atoms, min_layers=TAG_MIN_LAYERS, log=None) -> np.ndarray:
    """OC20-style surface tags: 1 = surface (free), 0 = bulk (frozen).

    An atom is tagged 1 if ANY of:

      1. it lies within TAG_HEIGHT_WINDOW of the topmost atom — OC20's height
         criterion, compared in Cartesian z with a tolerance instead of upstream's
         strict `<` on scaled coordinates, which is a coin flip when the interlayer
         spacing equals the window;
      2. a hydrogen-sized probe can reach it from above — approximates the property
         `is_adsorbate_intercalated` actually tests, and measured over all inputs it is
         the load-bearing criterion (183 structures fail the invariant without it, 0
         with it). NOT a guarantee: the probe is directional (+z) while the filter is a
         distance check in every direction, so undercuts, lateral H migration during
         relaxation, and post-tagging reconstruction can still produce a false
         rejection. Checked against the filter's exact criterion the residual gap is
         small (no violations at zero tolerance; 2 atoms on Mo2C(111) at 0.4 Å) but it
         is not zero;
      3. it is in the top `min_layers` atomic layers — a spacing-independent floor, so
         the freed fraction no longer depends on the material's interlayer distance.
         This is what fixes Mo2N specifically (its pit floors sit in layer 2), though
         corpus-wide it fixes little on its own (202 -> 183 structures).

    OC20's third criterion — under-coordination relative to bulk — was implemented and
    then removed: measured across all inputs it freed 115 extra atoms in 18 structures
    while fixing zero invariant violations, and it inherits the exact weakness that made
    upstream's Voronoi test useless on Mo2N (a vacancy-riddled interior already contains
    low-CN atoms, so pit atoms never look anomalous). Recover from git if ever needed.

    Stacks of TAG_THIN_LAYERS layers or fewer are freed entirely: there is no interior
    for frozen atoms to stand in for. This is counted in layers rather than Ångströms
    because an Å threshold cannot separate a true monolayer from a thin bulk-cut slab.

    Raises RuntimeError if the invariant is violated — better a loud failure than a
    slab whose every H placement is silently discarded.
    """
    z = atoms.positions[:, 2]
    tags = np.zeros(len(atoms), dtype=int)
    layers = layer_groups(z)

    # Count layers, not Ångströms. An Å threshold cannot tell a genuine monolayer from a
    # thin bulk-cut slab — MoB(110) is 8 atomic layers in under 8 Å and must be treated
    # as a normal slab, while graphene is one layer and MoS2 is three.
    if len(layers) <= TAG_THIN_LAYERS:
        if log:
            log.info(f"  tagging: {len(layers)} atomic layer(s) — no interior to hold as "
                     f"bulk, freeing all {len(atoms)} atoms")
        return np.ones(len(atoms), dtype=int)

    by_height = z >= z.max() - TAG_HEIGHT_WINDOW - 1e-6

    reachable = exposed_from_above(atoms)
    by_probe = np.zeros(len(atoms), dtype=bool)
    if reachable:
        by_probe[list(reachable)] = True

    by_layer = np.zeros(len(atoms), dtype=bool)
    for group in layers[:min_layers]:
        by_layer[group] = True

    tags[by_height | by_probe | by_layer] = 1

    # Criterion 3 makes this true by construction, so a failure means a bug in this
    # function rather than a bad structure.
    still_frozen = sorted(i for i in reachable if tags[i] == 0)
    if still_frozen:
        raise RuntimeError(
            f"surface_tags: {len(still_frozen)} probe-reachable atoms would stay frozen "
            f"(indices {still_frozen[:8]}…). Every H placement over them would be "
            f"rejected as adsorbate_intercalated."
        )

    if log:
        n_free = int(tags.sum())
        n_layers = sum(1 for g in layers if tags[g[0]] == 1)
        only_probe = int((by_probe & ~(by_height | by_layer)).sum())
        log.info(f"  tagging: {n_free}/{len(atoms)} atoms free across ~{n_layers} "
                 f"layer(s); {only_probe} freed only because a probe reaches them")
        if n_free > TAG_OVERFREE_WARN * len(atoms):
            log.warning(f"  tagging: freeing {100 * n_free / len(atoms):.0f}% of a "
                        f"{len(layers)}-layer slab — check the geometry")
    return tags


def tag_and_constrain(atoms, log=None):
    """Apply `surface_tags` to `atoms` and freeze everything tagged 0. Returns `atoms`."""
    from ase.constraints import FixAtoms

    tags = surface_tags(atoms, log=log)
    atoms.set_tags(tags.tolist())
    atoms.set_constraint(FixAtoms(mask=[t == 0 for t in tags]))
    return atoms


def frozen_but_reachable(atoms) -> list:
    """Indices of tag-0 atoms a hydrogen probe can reach — should always be empty.

    Post-relaxation diagnostic: tags are assigned to the *unrelaxed* geometry, so a
    surface that reconstructs during relaxation can expose an atom that was legitimately
    buried at the moment it was frozen.
    """
    tags = atoms.get_tags()
    return sorted(i for i in exposed_from_above(atoms) if tags[i] == 0)
