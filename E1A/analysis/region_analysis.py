"""E1A: v_x error by contact region, original against modified.

Three regions, applied to each condition's own trajectory on the eligible
analysis rows:

  near recontact   within +-5 ms of a geometric recontact
  shallow contact  gap <= 0, penetration 0-1 mm, and more than 20 ms from
                   every separation and recontact
  free flight      gap > 0, and more than 20 ms from every separation
                   and recontact

Calculates the balanced regional RMSE and its percentage change.

1. Base Metric: Computes RMSE_{j,R} per episode (j) and region (R) using 
   the de-normalized one-step v_x error (in mm/s). The sample size 
   N_{j,R} is tracked concurrently.
2. Aggregation: Averages episodes with equal weighting to find the 
   balanced regional RMSE: 
   RMSE_{R,balanced} = sqrt( mean_j(RMSE_{j,R}^2) )
3. Comparison: Calculates the percentage change as:
   change = 100 * ((RMSE_modified / RMSE_original) - 1)

A negative change indicates an improvement; a positive change 
indicates a deterioration.

    python E1A/analysis/region_analysis.py --seed 0
    python E1A/analysis/region_analysis.py --seed 0 --split test

The report is also saved to E1A/analysis/seed<N>/region_analysis.txt
E1A/analysis/test_split/seed<N>/ with --split test.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from diagnose import (RUN_ROOT, DEFAULT_SEED, load_bundles, episode_families,
                      log_to, seed_output_dir, add_split_argument)
from recontact_zoom import episode_signals
from window_analysis import CONDITIONS, PRIMARY_WINDOW_S, geometric_recontacts

GUARD_S = 0.020
SHALLOW_MM = 1.0

REGIONS = ("near_recontact", "shallow_contact", "free_flight")
REGION_LABELS = {
    "near_recontact": "Near recontact",
    "shallow_contact": "Shallow contact, away from transitions",
    "free_flight": "Free flight, away from transitions",
}


def transition_times(identifier, condition):
    """
    Every geometric recontact (gap + -> <= 0) and separation (gap <= 0 -> +)
    in this condition's full trace. Not clipped to the analysis span, so a
    transition just outside it still guards the rows next to it.
    """
    with np.load(RUN_ROOT / condition / "episodes" / identifier / "trace.npz") as data:
        time, gap = data["time"], data["gap"]

    recontacts = np.flatnonzero((gap[:-1] > 0) & (gap[1:] <= 0)) + 1
    separations = np.flatnonzero((gap[:-1] <= 0) & (gap[1:] > 0)) + 1
    return np.sort(time[np.concatenate((recontacts, separations))])


def region_masks(signals, recontacts, transitions, half_width_s, guard_s, shallow_mm):
    """Boolean row masks, one per region, on this condition's own rows."""
    time = signals["time"]
    gap = signals["gap"]  # mm, at t_i

    near = np.zeros(len(time), dtype=bool)
    for event in recontacts:
        near |= np.abs(time - event) <= half_width_s

    if len(transitions):
        distance = np.abs(time[:, None] - transitions[None, :]).min(axis=1)
        away = distance > guard_s
    else:
        away = np.ones(len(time), dtype=bool)

    contact = gap <= 0
    return {
        "near_recontact": near,
        "shallow_contact": contact & (gap >= -shallow_mm) & away,
        "free_flight": ~contact & away,
    }


def region_rmse(error, mask):
    """(RMSE in mm/s, N); RMSE is nan when the region is empty."""
    n = int(mask.sum())
    rmse = float(np.sqrt(np.mean(error[mask] ** 2))) if n else np.nan
    return rmse, n


def percent_change(original, modified):
    return 100.0 * (modified / original - 1.0)


def collect(bundles, episodes, args):
    """results[region][identifier][condition] = (rmse, n)"""
    results = {region: {} for region in REGIONS}

    for identifier in episodes:
        for condition in CONDITIONS:
            signals = episode_signals(*bundles[condition], identifier, condition)
            recontacts = geometric_recontacts(
                identifier, condition, signals["time"][0], signals["time"][-1]
            )
            masks = region_masks(
                signals, recontacts, transition_times(identifier, condition),
                args.window, args.guard, args.shallow,
            )
            for region, mask in masks.items():
                results[region].setdefault(identifier, {})[condition] = region_rmse(
                    signals["vx_error"], mask
                )

    return results


def balanced(per_episode, condition):
    """Episode-balanced RMSE over the episodes where both conditions have samples."""
    usable = [
        values for values in per_episode.values()
        if all(values[c][1] > 0 for c in CONDITIONS)
    ]
    if not usable:
        return np.nan, 0
    return float(np.sqrt(np.mean([v[condition][0] ** 2 for v in usable]))), len(usable)


def print_report(results, episodes, args):
    fmt = lambda v, w, p: f"{'n/a':>{w}}" if np.isnan(v) else f"{v:>{w}.{p}f}"

    print(f"\n{'=' * 86}")
    print(f"REGION ANALYSIS   near: +/-{args.window * 1e3:.0f} ms   "
          f"guard: {args.guard * 1e3:.0f} ms   shallow: 0-{args.shallow:g} mm"
          "      v_x error in mm/s")
    print("=" * 86)

    summary = []
    for region in REGIONS:
        per_episode = results[region]
        print(f"\n[{REGION_LABELS[region]}]")
        print(f"  {'episode':<26}{'N orig':>9}{'RMSE orig':>12}"
              f"{'N mod':>9}{'RMSE mod':>12}{'change %':>11}")

        for identifier in episodes:
            (a, n_a), (b, n_b) = (per_episode[identifier][c] for c in CONDITIONS)
            change = percent_change(a, b) if n_a and n_b else np.nan
            print(f"  {identifier:<26}{n_a:>9}{fmt(a, 12, 6)}"
                  f"{n_b:>9}{fmt(b, 12, 6)}{fmt(change, 11, 1)}")

        (a, used), (b, _) = (balanced(per_episode, c) for c in CONDITIONS)
        change = percent_change(a, b) if used else np.nan
        note = "" if used == len(episodes) else f"   ({used}/{len(episodes)} episodes had samples)"
        print(f"  {'balanced':<26}{'':>9}{fmt(a, 12, 6)}"
              f"{'':>9}{fmt(b, 12, 6)}{fmt(change, 11, 1)}{note}")
        summary.append((region, a, b, change))

    print(f"\n{'=' * 86}")
    print("SUMMARY   episode-balanced v_x RMSE (mm/s); negative change = improvement")
    print("=" * 86)
    print(f"  {'region':<42}{'original':>12}{'modified':>12}{'change %':>11}")
    for region, a, b, change in summary:
        print(f"  {REGION_LABELS[region]:<42}{fmt(a, 12, 6)}{fmt(b, 12, 6)}{fmt(change, 11, 1)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=float, default=PRIMARY_WINDOW_S,
                        help="near-recontact half width, s")
    parser.add_argument("--guard", type=float, default=GUARD_S,
                        help="exclusion around every transition, s")
    parser.add_argument("--shallow", type=float, default=SHALLOW_MM,
                        help="maximum penetration for shallow contact, mm")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Which training_runs/seed<N>/ to evaluate.")
    add_split_argument(parser)
    args = parser.parse_args()

    with log_to(seed_output_dir(args.seed, args.split) / "region_analysis.txt"):
        run(args)


def run(args):
    bundles = load_bundles(args.seed, args.split)

    families = episode_families()
    episodes = [
        Path(p).parent.name for p in bundles["original"][0].episode_paths
        if families.get(Path(p).parent.name) == "a3"
    ]

    results = collect(bundles, episodes, args)
    print_report(results, episodes, args)


if __name__ == "__main__":
    main()
