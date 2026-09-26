"""E1A: original against modified, summarized over every training seed.

Reported Metrics & Variability Summary:

  original, modified   Mean (min-max) across seeds of the episode-balanced
                       regional v_x RMSE (computed via region_analysis.py).
  per-seed change      Paired difference computed within each seed (original vs
                       modified), reported as median (min-max). Medians are used
                       because percentage changes are asymmetric (-50% and +100%
                       do not cancel out) and cannot be averaged directly.
  seeds                Count of seeds moving in the direction of the median change.

  Regions              Near recontact (+/- 5 ms), shallow contact (0-1 mm, > 20 ms from transition),
                       free flight (> 20 ms from transition)

The figure is the recontact close-up with every seed's prediction
overlaid, so seed-to-seed spread is visible next to the error itself.

    python E1A/analysis/seed_summary.py
    python E1A/analysis/seed_summary.py --seeds 0 1 2 3 42
    python E1A/analysis/seed_summary.py --split test

Output goes to E1A/analysis/summary/ 
E1A/analysis/test_split/summary/ with --split test.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from diagnose import (available_seeds, load_bundles, episode_families,
                      log_to, split_output_dir, add_split_argument)
from recontact_zoom import (CONDITIONS, INK, INK_MUTED, episode_signals,
                            recontact_times, around_event)
from region_analysis import (REGIONS, REGION_LABELS, GUARD_S, SHALLOW_MM,
                             collect, balanced, percent_change)
from window_analysis import PRIMARY_WINDOW_S

SEED_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#4a3aa7", "#e34948")


def seed_colors(seeds):
    everything = sorted(set(available_seeds()) | set(seeds))
    if len(everything) > len(SEED_COLORS):
        raise SystemExit(
            f"{len(everything)} seeds but only {len(SEED_COLORS)} colours."
        )
    return {seed: SEED_COLORS[everything.index(seed)] for seed in seeds}


def transition_episodes(bundles):
    families = episode_families()
    return [
        Path(p).parent.name for p in bundles["original"][0].episode_paths
        if families.get(Path(p).parent.name) == "a3"
    ]


def region_per_seed(bundles, episodes, args):
    """region -> (original, modified, change %), for one seed"""
    results = collect(bundles, episodes, args)
    out = {}
    for region in REGIONS:
        (a, used), (b, _) = (balanced(results[region], c) for c in CONDITIONS)
        out[region] = (a, b, percent_change(a, b) if used else np.nan)
    return out


def print_per_seed(per_seed, seeds):
    fmt = lambda v, w, p: f"{'n/a':>{w}}" if np.isnan(v) else f"{v:>{w}.{p}f}"

    print(f"\n{'=' * 86}")
    print("PER SEED   episode-balanced v_x RMSE (mm/s); negative change = improvement")
    print("=" * 86)
    for region in REGIONS:
        print(f"\n[{REGION_LABELS[region]}]")
        print(f"  {'seed':<8}{'original':>12}{'modified':>12}{'change %':>11}")
        for seed in seeds:
            a, b, change = per_seed[seed][region]
            print(f"  {seed:<8}{fmt(a, 12, 6)}{fmt(b, 12, 6)}{fmt(change, 11, 1)}")


def print_summary(per_seed, seeds):
    n = len(seeds)

    print(f"\n{'=' * 108}")
    print(f"SUMMARY OVER {n} SEEDS ({', '.join(map(str, seeds))})")
    print("  original, modified: mean (min-max) of episode-balanced v_x RMSE, mm/s")
    print("  per-seed change:    median (min to max) of the change within each seed")
    print("=" * 108)
    print(f"  {'region':<42}{'original':<24}{'modified':<24}"
          f"{'per-seed change':<28}{'seeds':<14}")

    for region in REGIONS:
        rows = np.array([per_seed[seed][region] for seed in seeds])
        original, modified, change = rows.T
        valid = change[~np.isnan(change)]

        spread = lambda v: f"{np.mean(v):.4f} ({np.min(v):.4f}-{np.max(v):.4f})"
        if len(valid):
            median = np.median(valid)
            span = (f"{median:+.1f}% ({valid.min():+.1f}% to "
                    f"{valid.max():+.1f}%)")
            if median <= 0:
                count = f"{int((valid < 0).sum())}/{n} improved"
            else:
                count = f"{int((valid > 0).sum())}/{n} worse"
        else:
            span, count = "n/a", f"0/{n} had samples"

        print(f"  {REGION_LABELS[region]:<42}{spread(original):<24}"
              f"{spread(modified):<24}{span:<28}{count:<14}")


def plot_overlay(windows, seeds, colors, identifier, cycle, destination):
    """
    One column per condition, shared y within a row. Truth, force and gap
    come from the data and are the same for every seed; only the model
    rows carry one curve per seed.
    """
    rows = [
        ("delta", "$\\Delta v_x$ over step (mm/s)"),
        ("vx_error", "signed $v_x$ error (mm/s)"),
        ("normal_force", "normal force at $t_i$ (N)"),
        ("gap", "signed gap at $t_i$ (mm)"),
    ]

    fig, axes = plt.subplots(
        len(rows), len(CONDITIONS), figsize=(13, 11),
        sharex=True, sharey="row", squeeze=False, layout="constrained",
    )

    for column, condition in enumerate(CONDITIONS):
        per_seed = windows[condition]
        reference = per_seed[seeds[0]]

        for row, (key, label) in enumerate(rows):
            axis = axes[row][column]
            axis.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
            axis.axhline(0, color=INK_MUTED, linewidth=0.8, alpha=0.6, zorder=1)

            if key == "delta":
                for seed in seeds:
                    axis.plot(per_seed[seed]["tau"], per_seed[seed]["vx_predicted"],
                              color=colors[seed], linewidth=1.0, alpha=0.9,
                              label=f"seed {seed}", zorder=2)
                axis.plot(reference["tau"], reference["vx_true"], color=INK,
                          linewidth=1.2, marker="o", markersize=2.5,
                          label="true", zorder=3)
            elif key == "vx_error":
                for seed in seeds:
                    axis.plot(per_seed[seed]["tau"], per_seed[seed]["vx_error"],
                              color=colors[seed], linewidth=1.0, alpha=0.9,
                              zorder=2)
            else:
                axis.plot(reference["tau"], reference[key], color=INK,
                          linewidth=0.7, marker="o", markersize=2.5, zorder=2)

            if column == 0:
                axis.set_ylabel(label, fontsize=9)
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=8)

        axes[0][column].set_title(
            f"{condition}   recontact at t = {reference['event_time']:.6f} s",
            fontsize=10,
        )
        axes[-1][column].set_xlabel(
            r"$\tau = t_i - t_{\mathrm{recontact}}$  (ms), each on its own event",
            fontsize=9,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", frameon=False,
               fontsize=9)
    fig.suptitle(f"{identifier}  --  matched cycle {cycle}, raw samples, "
                 f"{len(seeds)} seeds overlaid", fontsize=12)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure to:", destination)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Defaults to every training_runs/seed<N>/.")
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--plot-halfwidth", type=float, default=0.010,
                        help="Seconds either side of the event in the figure.")
    parser.add_argument("--window", type=float, default=PRIMARY_WINDOW_S,
                        help="near-recontact half width, s")
    parser.add_argument("--guard", type=float, default=GUARD_S,
                        help="exclusion around every transition, s")
    parser.add_argument("--shallow", type=float, default=SHALLOW_MM,
                        help="maximum penetration for shallow contact, mm")
    add_split_argument(parser)
    args = parser.parse_args()

    seeds = sorted(args.seeds) if args.seeds else available_seeds()
    if not seeds:
        raise SystemExit("No training_runs/seed<N>/ directories found.")
    colors = seed_colors(seeds)
    args.summary_dir = split_output_dir(args.split) / "summary"

    with log_to(args.summary_dir / "seed_summary.txt"):
        run(args, seeds, colors)


def run(args, seeds, colors):
    per_seed = {}
    # windows[identifier][condition][seed] -> close-up around the event
    windows = {}
    episodes = None

    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        bundles = load_bundles(seed, args.split)
        if episodes is None:
            episodes = transition_episodes(bundles)
        per_seed[seed] = region_per_seed(bundles, episodes, args)

        for identifier in episodes:
            for condition in CONDITIONS:
                events = recontact_times(identifier, condition)
                if args.cycle >= len(events):
                    raise SystemExit(f"{identifier}: no cycle {args.cycle}.")
                signals = episode_signals(*bundles[condition], identifier, condition)
                close = around_event(signals, events[args.cycle], args.plot_halfwidth)
                close["event_time"] = events[args.cycle]
                windows.setdefault(identifier, {}).setdefault(condition, {})[seed] = close

    print_per_seed(per_seed, seeds)
    print_summary(per_seed, seeds)

    print()
    for identifier in episodes:
        for condition in CONDITIONS:
            truths = [windows[identifier][condition][s]["vx_true"] for s in seeds]
            if not all(np.allclose(t, truths[0], atol=1e-6) for t in truths):
                raise SystemExit(f"{condition}/{identifier}: truth differs by seed.")
        plot_overlay(windows[identifier], seeds, colors, identifier, args.cycle,
                     args.summary_dir / "figures"
                     / f"recontact_{identifier}_cycle{args.cycle}_seeds.png")


if __name__ == "__main__":
    main()
