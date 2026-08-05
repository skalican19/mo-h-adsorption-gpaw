"""
scripts/adsorbml/3-extract_rank.py

AdsorbML step 3: For each slab, select the THERMODYNAMICALLY MOST STABLE H*
adsorption site (lowest ML adsorption energy, E_ads_ml), compute its
ΔG*H = E_ads_ml + 0.24 eV, and rank slabs by |ΔG*H| (Sabatier criterion).

CORRECTION (vs. previous version)
---------------------------------
The previous version selected, per slab, the candidate with the SMALLEST
|ΔG*H| (the placement closest to thermoneutral):

    best_idx = df["gibbs_free_ml_eV"].abs().idxmin()      # WRONG

Because each slab carries tens of AdsorbML placements spanning a wide energy
range, there is almost always *some* placement near ΔG≈0, so that rule drove
*every* material to ΔG≈0 by construction and made the slab ranking an artifact
of site cherry-picking rather than of surface chemistry.

The adsorption energy of a surface is defined by its most stable site, so the
correct per-slab representative is the minimum-energy configuration:

    best_idx = valid["E_ads_ml_eV"].idxmin()             # CORRECT

Slabs are then ranked across the dataset by |ΔG*H| (closeness to thermoneutral),
which is the legitimate Sabatier ranking once each slab's value is physical.

A physical sanity window [--emin, --emax] discards exploded/dissociated/
absorbed placements (e.g. E_ads ≈ -132 eV) that the AdsorbML 'anomalies' column
did not catch (that column is always empty by construction — run_adsorbml returns
only anomaly-free candidates; the rejected ones are in <slab>/anomalies.csv).
Every per-slab outcome is recorded with a quality_flag and the discarded raw
extremum, so nothing is hidden.

RELAXATION QUALITY IS REPORTED, NOT FILTERED ON. The selection rule above is
unchanged, so a candidate whose relaxation never converged can still top the
ranking. Fmax_slab/Fmax_adslab, slab_converged/adslab_converged and the step
counts are carried into the output CSV so that can be checked; results predating
2026-08 have no such data and read as 'unknown', never as converged.

Reads:  data/adsorbml_results/<name>/candidates.csv
        data/adsorbml_manifest.csv  (for slab_file paths)
Writes: data/adsorbml_results/ranked_candidates_stable.csv   (NEW FILE;
        the original ranked_candidates.csv is left untouched)

Usage:
  python scripts/adsorbml/3-extract_rank.py
  python scripts/adsorbml/3-extract_rank.py --emin -2.5 --emax 2.5
  python scripts/adsorbml/3-extract_rank.py --select min_abs_gibbs   # reproduce the OLD (buggy) ranking
  python scripts/adsorbml/3-extract_rank.py --out /path/to/custom.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    OUT_DIR, MANIFEST_CSV, ENTROPY_CORRECTION, DEFAULT_EMIN, DEFAULT_EMAX,
    FMAX_CONVERGED, fmax_from_traj, write_atomic_csv,
)

DEFAULT_OUT = OUT_DIR / "ranked_candidates_stable.csv"


def _get(row, key) -> float:
    """Numeric field from a Series-like row, NaN if absent/missing/unparseable.

    Convergence data is absent for pre-2026-08 results (no columns at all), so every
    read must degrade to 'unknown' rather than to a default that reads as converged.
    """
    if row is None or key not in row:
        return float("nan")
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return float("nan")
    return value


def _flag(row, key):
    """Tri-state convergence flag: True / False / None (= unknown, never 'converged')."""
    if row is None or key not in row:
        return None
    value = row[key]
    if isinstance(value, str):
        return {"true": True, "false": False}.get(value.strip().lower())
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return bool(value)


def _select_best(df, select, emin, emax):
    """Return (best_row, quality_flag, raw_extremum, n_all, n_in_window).

    select='stable'         -> lowest E_ads within [emin, emax] (CORRECT)
    select='min_abs_gibbs'  -> smallest |ΔG*H| over all candidates (OLD behaviour)
    """
    n_all = len(df)

    if select == "min_abs_gibbs":
        # Faithful reproduction of the old rule: no sanity window.
        idx = df["gibbs_free_ml_eV"].abs().idxmin()
        return df.loc[idx], "legacy_min_abs_gibbs", float("nan"), n_all, n_all

    # select == 'stable'
    raw_min = float(df["E_ads_ml_eV"].min())
    in_window = df[(df["E_ads_ml_eV"] >= emin) & (df["E_ads_ml_eV"] <= emax)]
    n_in_window = len(in_window)

    if n_in_window == 0:
        return None, "no_physical_candidate", raw_min, n_all, 0

    idx = in_window["E_ads_ml_eV"].idxmin()
    # Flag slabs where the unfiltered global minimum had to be discarded.
    flag = "ok" if np.isclose(in_window.loc[idx, "E_ads_ml_eV"], raw_min) \
        else "discarded_unphysical_min"
    return in_window.loc[idx], flag, raw_min, n_all, n_in_window


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", choices=["stable", "min_abs_gibbs"], default="stable",
                        help="Per-slab site selection rule (default: stable = most stable site).")
    parser.add_argument("--emin", type=float, default=DEFAULT_EMIN,
                        help=f"Lower E_ads sanity bound in eV (default {DEFAULT_EMIN}).")
    parser.add_argument("--emax", type=float, default=DEFAULT_EMAX,
                        help=f"Upper E_ads sanity bound in eV (default {DEFAULT_EMAX}).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output CSV path (default {DEFAULT_OUT.name}).")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_CSV).set_index("slab_name")

    rows = []
    skipped = []

    for candidates_csv in sorted(OUT_DIR.glob("*/candidates.csv")):
        slab_name = candidates_csv.parent.name

        try:
            df = pd.read_csv(candidates_csv)
        except Exception as exc:
            skipped.append((slab_name, str(exc)))
            continue

        if df.empty or "E_ads_ml_eV" not in df.columns:
            skipped.append((slab_name, "empty or missing E_ads_ml_eV"))
            continue

        df = df.dropna(subset=["E_ads_ml_eV"])
        if df.empty:
            skipped.append((slab_name, "all E_ads_ml_eV are NaN"))
            continue

        df["gibbs_free_ml_eV"] = df["E_ads_ml_eV"] + ENTROPY_CORRECTION

        best, flag, raw_min, n_all, n_in_window = _select_best(df, args.select, args.emin, args.emax)
        if best is None:
            skipped.append((slab_name, f"{flag} (raw min E_ads={raw_min:.2f} eV, "
                                       f"window [{args.emin}, {args.emax}])"))
            continue

        # The adsorbate is a single H (*H); locate it by element rather than
        # assuming it is the last atom. Take the last H if several are present.
        try:
            adslab = read(str(best["traj_path"]))
            h_idx = [i for i, a in enumerate(adslab) if a.symbol == "H"]
            if not h_idx:
                print(f"  WARN {slab_name}: no H atom in adslab; H position set to NaN")
                h_pos = [float("nan")] * 3
            else:
                h_pos = adslab[h_idx[-1]].position
        except Exception as exc:
            print(f"  WARN {slab_name}: could not read H position: {exc}")
            h_pos = [float("nan")] * 3

        mrow = manifest.loc[slab_name] if slab_name in manifest.index else None
        slab_file = mrow["slab_file"] if mrow is not None else ""

        # Relaxation quality. Prefer the columns step 2 / step 1 recorded (free) over
        # re-reading a traj per slab; fall back to fmax_from_traj for pre-2026-08 data,
        # which has neither the columns nor forces in the candidate trajs.
        fmax_adslab = _get(best, "Fmax_adslab_eV_per_Ang")
        if np.isnan(fmax_adslab):
            fmax_adslab = fmax_from_traj(best["traj_path"])
        fmax_slab = _get(mrow, "relax_fmax")
        if np.isnan(fmax_slab):
            fmax_slab = fmax_from_traj(slab_file)

        rows.append({
            "slab_name":         slab_name,
            "best_rank":         int(best["candidate_rank"]),
            "E_ads_ml_eV":       float(best["E_ads_ml_eV"]),
            "gibbs_free_ml_eV":  float(best["gibbs_free_ml_eV"]),
            "h_x":               float(h_pos[0]),
            "h_y":               float(h_pos[1]),
            "h_z":               float(h_pos[2]),
            "slab_file":         slab_file,
            "candidate_file":    str(best["traj_path"]),
            # Relaxation quality — reported, NOT filtered on. The selection rule is
            # still lowest E_ads in the sanity window, so an unconverged site can top
            # the ranking; sort by these before trusting the top of the list.
            "Fmax_slab_eV_per_Ang":   fmax_slab,
            "Fmax_adslab_eV_per_Ang": fmax_adslab,
            "slab_converged":    _flag(mrow, "relax_converged"),
            "slab_nsteps":       _get(mrow, "relax_nsteps"),
            "adslab_converged":  _flag(best, "adslab_converged"),
            "adslab_nsteps":     _get(best, "adslab_nsteps"),
            # diagnostics (extra columns; ignored by downstream loaders)
            "n_candidates":      n_all,
            "n_in_window":       n_in_window,
            "E_ads_raw_min_eV":  raw_min,
            "quality_flag":      flag,
        })

    if not rows:
        print("No results found. Run 2-run_adsorbml.py first.")
        return

    ranked = (pd.DataFrame(rows)
              .sort_values("gibbs_free_ml_eV", key=lambda s: s.abs(), na_position="last")
              .reset_index(drop=True))
    write_atomic_csv(ranked, args.out)
    print(f"\nRanked candidates written: {args.out}  ({len(ranked)} slabs, select='{args.select}')")

    if skipped:
        print(f"\nSkipped {len(skipped)} slabs:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    flag_counts = ranked["quality_flag"].value_counts().to_dict()
    print(f"\nQuality flags: {flag_counts}")

    # Convergence report. Nothing above filtered on it — this is what tells you how
    # much of the ranking rests on non-stationary geometries.
    n = len(ranked)
    print("\nRelaxation quality (reported, NOT filtered on):")
    for label, fmax_col, flag_col in (("slab  ", "Fmax_slab_eV_per_Ang", "slab_converged"),
                                      ("adslab", "Fmax_adslab_eV_per_Ang", "adslab_converged")):
        filled  = int(ranked[fmax_col].notna().sum())
        by_fmax = int((ranked[fmax_col] <= FMAX_CONVERGED).sum())
        flags   = ranked[flag_col]
        yes     = int((flags == True).sum())          # noqa: E712 — None must not count
        no      = int((flags == False).sum())         # noqa: E712
        unknown = n - yes - no
        print(f"  {label}: fmax recorded {filled}/{n}; ≤{FMAX_CONVERGED} eV/Å {by_fmax}/{n}  |  "
              f"flags: converged {yes}, not converged {no}, unknown {unknown}")
    if (ranked["adslab_converged"] == False).any():   # noqa: E712
        worst = ranked.nlargest(3, "Fmax_adslab_eV_per_Ang")[
            ["slab_name", "Fmax_adslab_eV_per_Ang"]]
        print("  highest adslab fmax in the ranking:")
        print(worst.to_string(index=False, header=False))

    print(f"\nTop 20 by |ΔG*H| (most-stable site, select='{args.select}'):")
    cols = ["slab_name", "gibbs_free_ml_eV", "E_ads_ml_eV", "best_rank",
            "Fmax_slab_eV_per_Ang", "Fmax_adslab_eV_per_Ang", "adslab_converged",
            "quality_flag"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(ranked.head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
