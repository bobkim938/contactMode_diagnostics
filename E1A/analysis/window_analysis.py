"""E1A task 4: prediction error near recontact against error away from it.

The +-5 ms window is the primary one, frozen before any of this was run;
the other widths are declared here as predefined sensitivities, not
chosen after seeing results.

Two traps this report is built to avoid:

  * Concentration can fall because the BACKGROUND got worse, which is not
    an improvement. Absolute RMSE inside and outside are therefore always
    printed side by side, and a rise in background error is flagged.
  * A lower peak with a wider skirt is not the same as less error
    everywhere. The window sweep separates those: an improvement that
    survives at +-20 ms is a real reduction, one that only shows at +-2 ms
    means the error moved outward rather than went away.

Each condition uses its OWN geometric recontact instants, since the two
do not re-touch at the same moment.

    python E1A/analysis/window_analysis.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from diagnose import (RUN_ROOT, init_model, load_dataset, compute_predictions,
                      episode_families, FAMILY_ORDER, FAMILY_LABELS)
from recontact_zoom import episode_signals

PRIMARY_WINDOW_S = 0.005
SENSITIVITY_WINDOWS_S = (0.002, 0.005, 0.010, 0.020)
CONDITIONS = ("original", "modified")


def geometric_recontacts(identifier, condition, time_lo, time_hi):
    """
    Gap sign changes (+ -> <= 0) in this condition's own trace, kept to
    the span the analysis rows actually cover. Controls never separate,
    so they correctly yield none.
    """
    with np.load(RUN_ROOT / condition / "episodes" / identifier / "trace.npz") as data:
        time, gap = data["time"], data["gap"]

    crossings = time[np.flatnonzero((gap[:-1] > 0) & (gap[1:] <= 0)) + 1]
    return crossings[(crossings >= time_lo) & (crossings <= time_hi)]


def window_metrics(signals, events, half_width_s):
    """The five reported quantities, plus the pieces needed to read them."""
    time = signals["time"]
    error = signals["vx_error"]
    squared = error ** 2

    inside = np.zeros(len(time), dtype=bool)
    for event in events:
        inside |= np.abs(time - event) <= half_width_s

    total_squared = squared.sum()
    outside = ~inside

    return {
        "n_events": len(events),
        "n_inside": int(inside.sum()),
        "n_total": int(len(time)),
        "sample_fraction": float(inside.mean()),
        "rmse_inside": float(np.sqrt(squared[inside].mean())) if inside.any() else np.nan,
        "rmse_outside": float(np.sqrt(squared[outside].mean())) if outside.any() else np.nan,
        "error_fraction": float(squared[inside].sum() / total_squared) if total_squared else np.nan,
        "max_abs_inside": float(np.abs(error[inside]).max()) if inside.any() else np.nan,
        "rmse_all": float(np.sqrt(squared.mean())),
        "concentration": (float(squared[inside].sum() / total_squared / inside.mean())
                          if inside.any() and total_squared else np.nan),
    }


def collect(bundles, identifier):
    """Signals and this-condition recontact instants, per condition."""
    out = {}
    for condition in CONDITIONS:
        signals = episode_signals(*bundles[condition], identifier, condition)
        events = geometric_recontacts(
            identifier, condition, signals["time"][0], signals["time"][-1]
        )
        out[condition] = (signals, events)
    return out


def print_primary(bundles, episodes, families, half_width_s):
    print(f"\n{'=' * 118}")
    print(f"PRIMARY WINDOW  +/-{half_width_s * 1e3:.0f} ms   (frozen before testing)"
          "      v_x error in mm/s")
    print("=" * 118)

    for family in FAMILY_ORDER:
        members = [e for e in episodes if families.get(e) == family]
        if not members:
            continue
        print(f"\n[{family}]  {FAMILY_LABELS.get(family, '')}")
        print(f"  {'episode':<26}{'cond':>10}{'ev':>4}{'RMSE in':>11}{'RMSE out':>11}"
              f"{'samp %':>9}{'err %':>9}{'conc':>8}{'max|e| in':>12}{'RMSE all':>11}")

        for identifier in members:
            data = collect(bundles, identifier)
            rows = {}
            for condition in CONDITIONS:
                signals, events = data[condition]
                m = window_metrics(signals, events, half_width_s)
                rows[condition] = m
                fmt = lambda v, w, p: (f"{'n/a':>{w}}" if np.isnan(v) else f"{v:>{w}.{p}f}")
                print(f"  {identifier:<26}{condition:>10}{m['n_events']:>4}"
                      f"{fmt(m['rmse_inside'], 11, 6)}{fmt(m['rmse_outside'], 11, 6)}"
                      f"{m['sample_fraction'] * 100:>9.2f}"
                      f"{m['error_fraction'] * 100:>9.2f}" if not np.isnan(m['error_fraction'])
                      else f"  {identifier:<26}{condition:>10}{m['n_events']:>4}"
                           f"{fmt(m['rmse_inside'], 11, 6)}{fmt(m['rmse_outside'], 11, 6)}"
                           f"{m['sample_fraction'] * 100:>9.2f}{'n/a':>9}", end="")
                print(f"{fmt(m['concentration'], 8, 1)}{fmt(m['max_abs_inside'], 12, 6)}"
                      f"{m['rmse_all']:>11.6f}")

            a, b = rows["original"], rows["modified"]
            note = ""
            if b["rmse_outside"] > a["rmse_outside"]:
                note = "   <- BACKGROUND WORSE: read concentration with care"
            print(f"  {'':<26}{'delta':>10}{'':>4}"
                  f"{b['rmse_inside'] - a['rmse_inside']:>+11.6f}"
                  f"{b['rmse_outside'] - a['rmse_outside']:>+11.6f}"
                  f"{'':>9}{'':>9}"
                  f"{(b['concentration'] - a['concentration']) if not np.isnan(a['concentration']) else float('nan'):>+8.1f}"
                  f"{b['max_abs_inside'] - a['max_abs_inside']:>+12.6f}"
                  f"{b['rmse_all'] - a['rmse_all']:>+11.6f}{note}")


def print_sensitivity(bundles, episodes, families, widths):
    print(f"\n{'=' * 118}")
    print("PREDEFINED WINDOW SENSITIVITY      v_x error in mm/s")
    print("  An improvement that holds as the window widens is a real reduction;")
    print("  one that fades means the error moved outward rather than went away.")
    print("=" * 118)

    for identifier in [e for e in episodes if families.get(e) == "a3"]:
        data = collect(bundles, identifier)
        print(f"\n[{identifier}]")
        print(f"  {'window':>8}{'samp %':>9}"
              f"{'RMSE in orig':>14}{'RMSE in mod':>13}{'ratio':>8}"
              f"{'RMSE out orig':>15}{'RMSE out mod':>14}{'ratio':>8}"
              f"{'err % orig':>12}{'err % mod':>11}")

        for half_width_s in widths:
            m = {c: window_metrics(*data[c], half_width_s) for c in CONDITIONS}
            a, b = m["original"], m["modified"]
            flag = "  <-- primary" if abs(half_width_s - PRIMARY_WINDOW_S) < 1e-12 else ""
            print(f"  {half_width_s * 1e3:>6.0f}ms{a['sample_fraction'] * 100:>9.2f}"
                  f"{a['rmse_inside']:>14.6f}{b['rmse_inside']:>13.6f}"
                  f"{b['rmse_inside'] / a['rmse_inside']:>8.3f}"
                  f"{a['rmse_outside']:>15.6f}{b['rmse_outside']:>14.6f}"
                  f"{b['rmse_outside'] / a['rmse_outside']:>8.3f}"
                  f"{a['error_fraction'] * 100:>12.2f}{b['error_fraction'] * 100:>11.2f}{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=float, default=PRIMARY_WINDOW_S)
    parser.add_argument("--windows", type=float, nargs="+",
                        default=list(SENSITIVITY_WINDOWS_S))
    args = parser.parse_args()

    model_0, checkpoint_0, model_0_9, checkpoint_0_9 = init_model()
    bundles = {}
    for condition, model, checkpoint, is_modified in (
        ("original", model_0, checkpoint_0, False),
        ("modified", model_0_9, checkpoint_0_9, True),
    ):
        if checkpoint["config"]["variant"] != condition:
            raise SystemExit(
                f"Checkpoint says {checkpoint['config']['variant']!r} but was "
                f"loaded as {condition!r}."
            )
        stats = {name: tensor.detach().cpu().numpy()
                 for name, tensor in checkpoint["normalization_stats"].items()}
        dataset = load_dataset(stats, is_modified)
        predicted, target = compute_predictions(model, dataset, stats)
        bundles[condition] = (dataset, predicted, target)

    families = episode_families()
    episodes = [Path(p).parent.name for p in bundles["original"][0].episode_paths]

    print_primary(bundles, episodes, families, args.primary)
    print_sensitivity(bundles, episodes, families, args.windows)


if __name__ == "__main__":
    main()
