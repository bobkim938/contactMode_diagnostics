"""E1A: the recontact close-up, each condition on its own event clock.

Every quantity is plotted against

    tau = t_i - t_recontact

where t_recontact is that condition's own geometric recontact instant.

Figures show raw samples. A light moving average is drawn over them as a
guide, but every number reported here comes from the unsmoothed error.

    python E1A/analysis/recontact_zoom.py --seed 0

Figures and the printed metrics go to E1A/analysis/seed<N>/.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

E1A = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(E1A))

from diagnose import (RUN_ROOT, DEFAULT_SEED, load_bundles, episode_families,
                      log_to, seed_output_dir)

CONDITIONS = ("original", "modified")
COLORS = {"original": "#1f77b4", "modified": "#ff7f0e"}
INK = "black"
INK_MUTED = "gray"


def recontact_times(identifier, condition):
    """
    Geometric recontact instants, per cycle, from this condition's own
    episode.json -- never from the other condition's.

    entry_s is cross-checked against the gap sign change in the trace, so
    a metadata field that had drifted from the simulation would be caught
    here rather than silently shifting every tau.
    """
    episode_dir = RUN_ROOT / condition / "episodes" / identifier
    cycles = json.loads((episode_dir / "episode.json").read_text())["quality"]["cycles"]
    entries = [float(cycle["entry_s"]) for cycle in cycles]

    with np.load(episode_dir / "trace.npz") as data:
        time, gap = data["time"], data["gap"]
    crossings = time[np.flatnonzero((gap[:-1] > 0) & (gap[1:] <= 0)) + 1]

    for entry in entries:
        if not np.any(np.isclose(crossings, entry, rtol=0, atol=1e-9)):
            raise ValueError(
                f"{condition}/{identifier}: entry_s {entry} is not a gap "
                "sign change; metadata and trace disagree."
            )

    return entries


def episode_signals(dataset, predicted, target, identifier, condition):
    """
    Per-sample signals for one episode, on the rows that entered the
    analysis (analysis_mask & valid_transition), so they line up with the
    prediction error row for row.
    """
    identifiers = [Path(p).parent.name for p in dataset.episode_paths]
    index = identifiers.index(identifier)
    keep = dataset.episode_ids == index
    rows = dataset.row_indices[keep]

    with np.load(RUN_ROOT / condition / "episodes" / identifier / "trace.npz") as data:
        time = data["time"][rows]
        normal_force = data["normal_force"][rows]
        gap = data["gap"][rows] * 1e3  # mm

    predicted = predicted.detach().cpu().numpy()[keep]
    truth = target.detach().cpu().numpy()[keep]

    return {
        "time": time,
        "vx_true": truth[:, 2] * 1e3,          # mm/s
        "vx_predicted": predicted[:, 2] * 1e3,  # mm/s
        "vx_error": (predicted - truth)[:, 2] * 1e3,
        "normal_force": normal_force,           # N
        "gap": gap,                             # mm
    }


def around_event(signals, event_time, half_width_s):
    """Samples within +/- half_width of one event, carrying tau in ms."""
    offset = signals["time"] - event_time
    keep = np.abs(offset) <= half_width_s

    window = {name: values[keep] for name, values in signals.items()}
    window["tau"] = offset[keep] * 1e3
    return window


def moving_average(values, window):
    """A drawing aid only; no reported number is computed from this."""
    window = int(min(max(window, 1), len(values)))
    if window <= 1:
        return values.copy()
    left = window // 2
    padded = np.pad(values, (left, window - 1 - left), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def transient_metrics(window, threshold_factor=3.0, far_tau_ms=10.0):
    """
    The three questions, answered from the UNSMOOTHED signed error.

    baseline  RMS error far from the event: the model's usual level here.
    lead      how far before touching the error first leaves that level,
              which is the anticipation the E1 zoom exposed.
    recovery  how long after the event it stays out.
    """
    tau = window["tau"]
    error = window["vx_error"]

    far = np.abs(tau) >= far_tau_ms
    baseline = float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else float("nan")
    threshold = threshold_factor * baseline

    out = np.abs(error) > threshold
    before = out & (tau < 0)
    after = out & (tau >= 0)

    peak = int(np.argmax(np.abs(error)))
    onset = window["normal_force"] > 0

    return {
        "baseline_rms": baseline,
        "threshold": threshold,
        "lead_ms": float(-tau[before].min()) if before.any() else 0.0,
        "recovery_ms": float(tau[after].max()) if after.any() else 0.0,
        "peak_error": float(error[peak]),
        "peak_tau_ms": float(tau[peak]),
        "error_at_tau0": float(error[np.argmin(np.abs(tau))]),
        "onset_steps": int(np.flatnonzero(onset)[0] - np.searchsorted(tau, 0.0))
                       if onset.any() else -1,
        "peak_force": float(window["normal_force"].max()),
    }


def plot_episode(windows, identifier, cycle, destination, smooth_window=9):
    """Four aligned rows, one column per condition, shared y within a row."""
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
        window = windows[condition]
        colour = COLORS[condition]

        for row, (key, label) in enumerate(rows):
            axis = axes[row][column]
            axis.axvline(0, color=colour, linewidth=1.4, zorder=1)
            axis.axhline(0, color=INK_MUTED, linewidth=0.8, alpha=0.6, zorder=1)

            if key == "delta":
                axis.plot(window["tau"], window["vx_true"], color=INK,
                          linewidth=0.8, marker="o", markersize=2.5,
                          label="true", zorder=3)
                axis.plot(window["tau"], window["vx_predicted"], color=colour,
                          linewidth=0.8, marker="s", markersize=2.5,
                          label="predicted", zorder=2)
                if row == 0:
                    axis.legend(frameon=False, fontsize=8, loc="lower left")
            else:
                axis.plot(window["tau"], window[key], color=INK, linewidth=0.7,
                          marker="o", markersize=2.5, zorder=2)
                if key == "vx_error":
                    # Guide only; metrics use the raw samples above.
                    axis.plot(window["tau"],
                              moving_average(window[key], smooth_window),
                              color=colour, linewidth=1.8, alpha=0.85, zorder=3)

            if column == 0:
                axis.set_ylabel(label, fontsize=9)
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=8)

        axes[0][column].set_title(
            f"{condition}   recontact at t = {windows[condition]['event_time']:.6f} s",
            fontsize=10,
        )
        axes[-1][column].set_xlabel(
            r"$\tau = t_i - t_{\mathrm{recontact}}$  (ms), each on its own event",
            fontsize=9,
        )

    fig.suptitle(f"{identifier}  --  matched cycle {cycle}, raw samples", fontsize=12)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure to:", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--plot-halfwidth", type=float, default=0.010,
                        help="Seconds either side of the event in the figure.")
    parser.add_argument("--metric-halfwidth", type=float, default=0.050,
                        help="Wider window, so the baseline has far-field samples.")
    parser.add_argument("--threshold-factor", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Which training_runs/seed<N>/ to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to E1A/analysis/seed<N>/figures.")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = seed_output_dir(args.seed) / "figures"

    with log_to(seed_output_dir(args.seed) / "recontact_zoom.txt"):
        run(args)


def run(args):
    bundles = load_bundles(args.seed)

    families = episode_families()
    dataset = bundles["original"][0]
    transition = [Path(p).parent.name for p in dataset.episode_paths
                  if families.get(Path(p).parent.name) == "a3"]
    print("\nTransition episodes:", transition)

    header = (f"  {'episode':<10}{'cyc':>4}{'condition':>11}{'t_event (s)':>14}"
              f"{'baseline':>11}{'lead ms':>9}{'recov ms':>10}"
              f"{'peak err':>10}{'peak tau':>10}{'err@0':>9}")
    print("\nMetrics from unsmoothed error; error in mm/s\n" + header)

    for identifier in transition:
        windows = {}
        for condition in CONDITIONS:
            signals = episode_signals(*bundles[condition], identifier, condition)
            events = recontact_times(identifier, condition)
            if args.cycle >= len(events):
                raise SystemExit(f"{identifier}: no cycle {args.cycle}.")
            event = events[args.cycle]

            wide = around_event(signals, event, args.metric_halfwidth)
            metrics = transient_metrics(wide, args.threshold_factor)

            close = around_event(signals, event, args.plot_halfwidth)
            close["event_time"] = event
            windows[condition] = close

            print(f"  {identifier:<10}{args.cycle:>4}{condition:>11}{event:>14.6f}"
                  f"{metrics['baseline_rms']:>11.6f}{metrics['lead_ms']:>9.3f}"
                  f"{metrics['recovery_ms']:>10.3f}{metrics['peak_error']:>+10.4f}"
                  f"{metrics['peak_tau_ms']:>+10.3f}{metrics['error_at_tau0']:>+9.4f}")

        shift = (windows["modified"]["event_time"]
                 - windows["original"]["event_time"]) * 1e3
        print(f"  {'':<10}{'':>4}{'event shift':>11}{shift:>+14.4f} ms")

        plot_episode(windows, identifier, args.cycle,
                     args.output_dir / f"recontact_{identifier}_cycle{args.cycle}.png")


if __name__ == "__main__":
    main()
