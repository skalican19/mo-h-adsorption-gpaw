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
absorbed placements (e.g. E_ads ≈ -132 eV) that the AdsorbML 'anomalies' field
did not catch (it is empty for the entire dataset). Every per-slab outcome is
recorded with a quality_flag and the discarded raw extremum, so nothing is
hidden.

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

        # H atom is always the last atom in the adslab trajectory
        try:
            adslab = read(str(best["traj_path"]))
            h_pos  = adslab[-1].position
        except Exception as exc:
            print(f"  WARN {slab_name}: could not read H position: {exc}")
            h_pos = [float("nan")] * 3

        slab_file = (
            manifest.loc[slab_name, "slab_file"]
            if slab_name in manifest.index else ""
        )

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
            "Fmax_slab_eV_per_Ang":   fmax_from_traj(slab_file),
            "Fmax_adslab_eV_per_Ang": fmax_from_traj(best["traj_path"]),
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

    n_slab_ok   = ranked["Fmax_slab_eV_per_Ang"].notna().sum()
    n_conv      = (ranked["Fmax_slab_eV_per_Ang"] <= FMAX_CONVERGED).sum()
    n_adslab_ok = ranked["Fmax_adslab_eV_per_Ang"].notna().sum()
    print(f"Fmax_slab filled: {n_slab_ok}/{len(ranked)}; converged (≤{FMAX_CONVERGED} eV/Å): "
          f"{n_conv}/{len(ranked)}")
    print(f"Fmax_adslab filled: {n_adslab_ok}/{len(ranked)} "
          f"(candidate trajs lack forces — re-run 2-run_adsorbml.py to populate)")

    print(f"\nTop 20 by |ΔG*H| (most-stable site, select='{args.select}'):")
    cols = ["slab_name", "gibbs_free_ml_eV", "E_ads_ml_eV", "best_rank",
            "Fmax_slab_eV_per_Ang", "quality_flag"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(ranked.head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
