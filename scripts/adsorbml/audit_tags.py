"""
scripts/adsorbml/audit_tags.py

Static audit of surface tagging across every input structure. CPU-only, no model,
no GPU — run this before spending GPU time on a campaign.

What it checks, per structure:
  - how many atoms and how many atomic layers the rule frees
  - how many frozen atoms a hydrogen-sized probe can still reach

The last column is the one that matters. A frozen atom H can touch makes every
placement over it a false `adsorbate_intercalated` rejection, and enough of them
means the slab silently yields zero candidates. See adsorbml/tagging.py.

Usage:
  python scripts/adsorbml/audit_tags.py
  python scripts/adsorbml/audit_tags.py --include "Mo2N_*,Ni_Mo2C_*"
  python scripts/adsorbml/audit_tags.py --compare      # new rule vs the old 2 Å rule
  python scripts/adsorbml/audit_tags.py --max-atoms 1200

Exit status is 1 if any structure fails the reachability invariant, so this can gate
a submission script.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import discover_structures
from adsorbml._common import DATA_INPUTS
from adsorbml.tagging import exposed_from_above, layer_groups, surface_tags


def legacy_tags(atoms) -> np.ndarray:
    """The pre-fix rule, kept only so --compare can show what changed."""
    z = atoms.positions[:, 2]
    return (z > z.max() - 2.0).astype(int)


def audit_one(atoms, rule) -> dict:
    tags = rule(atoms)
    reachable = exposed_from_above(atoms)
    z = atoms.positions[:, 2]
    layers = layer_groups(z)
    return {
        "natoms":     len(atoms),
        "free":       int(tags.sum()),
        "layers":     sum(1 for g in layers if tags[g[0]] == 1),
        "bad":        sum(1 for i in reachable if tags[i] == 0),
        "thickness":  float(z.max() - z.min()),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit AdsorbML surface tagging")
    parser.add_argument("--include", type=str, default=None,
                        help="Comma-separated glob patterns to filter structures")
    parser.add_argument("--compare", action="store_true",
                        help="Also report the legacy fixed-2 Å rule for contrast")
    parser.add_argument("--max-atoms", type=int, default=1200,
                        help="Skip structures larger than this (the probe raster is "
                             "O(atoms x grid); default 1200)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print failures and the summary")
    args = parser.parse_args()

    patterns = [p.strip() for p in args.include.split(",")] if args.include else None
    structures = discover_structures(DATA_INPUTS, include_patterns=patterns)

    failures, skipped, rows = [], [], []
    for _, _, path in sorted(structures, key=lambda s: s[2].name):
        name = path.name
        try:
            atoms = read(str(path / "POSCAR"))
        except Exception as exc:
            skipped.append((name, f"unreadable: {exc}"))
            continue
        if len(atoms) > args.max_atoms:
            skipped.append((name, f"{len(atoms)} atoms > --max-atoms"))
            continue

        try:
            new = audit_one(atoms, surface_tags)
        except RuntimeError as exc:
            failures.append((name, str(exc)))
            continue

        rows.append((name, new))
        if new["bad"]:
            failures.append((name, f"{new['bad']} frozen atoms are probe-reachable"))

        if not args.quiet:
            line = (f"{name:42s} n={new['natoms']:4d}  free={new['free']:4d}  "
                    f"layers={new['layers']:2d}  reachable-but-frozen={new['bad']:3d}")
            if args.compare:
                old = audit_one(atoms, legacy_tags)
                line += f"   | legacy: free={old['free']:4d} layers={old['layers']:2d} bad={old['bad']:3d}"
            print(line)

    print()
    print(f"Audited {len(rows)} structures  |  skipped {len(skipped)}")
    if rows:
        thin = sum(1 for _, r in rows if r["layers"] < 2)
        print(f"  freeing < 2 layers          : {thin}")
        print(f"  frozen-but-reachable atoms  : {sum(1 for _, r in rows if r['bad'])}")
        frac = [r["free"] / r["natoms"] for _, r in rows]
        print(f"  free fraction               : min {min(frac):.1%}  "
              f"median {sorted(frac)[len(frac) // 2]:.1%}  max {max(frac):.1%}")
    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for name, why in skipped[:10]:
            print(f"  {name}: {why}")
        if len(skipped) > 10:
            print(f"  … and {len(skipped) - 10} more")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for name, why in failures:
            print(f"  {name}: {why}")
        by_family = Counter(n.split("_(")[0] for n, _ in failures)
        print(f"\n  by family: {dict(by_family)}")
        return 1

    print("\nAll audited structures pass the reachability invariant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
