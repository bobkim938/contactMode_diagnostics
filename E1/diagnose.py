import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from model import dynamicsMLP
from dataset import E1Dataset

CHECKPOINT_DIR = (
    Path(__file__).resolve().parent / "training_runs" / "20260921_102817_303959"
)
# Outputs are grouped per checkpoint, e.g. analysis/20260921_102817_303959/
ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis" / CHECKPOINT_DIR.name

# Timing convention, used consistently by everything below: a sample is
# indexed by t_i, the START of the prediction interval [t_i, t_i+1]]

MODE_LABELS = {
    0: "Free",
    1: "Quasi-stick",
    2: "Boundary",
    3: "Slide",
    4: "Contact",
}

MODE_COLORS = {
    0: "#0072B2",  # Free        – blue
    1: "#009E73",  # Quasi-stick – green
    2: "#CC79A7",  # Boundary    – pink/purple
    3: "#D55E00",  # Slide       – vermillion (red-orange)
    4: "#7F7F7F",  # Contact     – neutral gray
}

INK = "black"
INK_MUTED = "gray"
SURFACE = "white"
WINDOW = "orange"

def init_model():
    model = dynamicsMLP()

    path = CHECKPOINT_DIR / "best.pt"

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded from:", path)
    print("Checkpoint epoch:", checkpoint.get("epoch", "not recorded"))

    return model, checkpoint

def load_dataset(norm_stats):
    dataset = E1Dataset(
        roots = Path(__file__).resolve().parent / "run_001",
        split = "validation",
        stats = norm_stats
    )
    return dataset

def compute_predictions(model, dataset, norm_stats, batch_size=1024):
    """
    de-normalized predicted and true one-step deltas, [N, 4] each, in
    dataset order: [delta_qx, delta_qy, delta_vx, delta_vy] in metres
    and m/s. Each row is the delta over [t_i, t_i+1], indexed by t_i
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    target_scale = torch.as_tensor(norm_stats["target_scale"])
    target_mean = torch.as_tensor(norm_stats["target_mean"])

    predicted_list = []
    target_list = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            predicted_list.append(model(inputs) * target_scale + target_mean)
            target_list.append(targets * target_scale + target_mean)

    return torch.cat(predicted_list, dim=0), torch.cat(target_list, dim=0)

def compute_errors(model, dataset, norm_stats, batch_size=1024):
    """
    de-normalized one-step prediction error, [N, 4], in dataset order:
    [delta_qx, delta_qy, delta_vx, delta_vy] in metres and m/s
    """
    predicted, target = compute_predictions(
        model, dataset, norm_stats, batch_size
    )
    return predicted - target

def _runs(mode, row_indices):
    """
    (start, stop, mode) for each maximal constant-mode run, also broken
    wherever the retained rows are not consecutive in the raw trace
    """
    breaks = np.flatnonzero(
        (np.diff(mode) != 0) | (np.diff(row_indices) != 1)
    ) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(mode)]))
    return [(int(a), int(b), int(mode[a])) for a, b in zip(starts, stops)]

def _rolling_rms(values, window):
    """Root mean square over a centred window, same length as values"""
    window = int(min(max(window, 1), len(values)))
    if window <= 1:
        return values.copy()

    left = window // 2
    padded = np.pad(values ** 2, (left, window - 1 - left), mode="edge")
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(padded, kernel, mode="valid"))

def collect_episode_errors(dataset, error_tensor):
    """
    Regroup the flat error tensor into one record per episode, paired with
    the trace diagnostics (time, contact mode, detected transitions) needed
    to read the error against the contact structure
    """
    errors = error_tensor.detach().cpu().numpy()
    episodes = []

    for episode_id, episode_path in enumerate(dataset.episode_paths):
        keep = dataset.episode_ids == episode_id
        rows = dataset.row_indices[keep]

        with np.load(dataset.root / episode_path) as data:
            time = data["time"][rows]
            mode = data["mode"][rows]

        events_path = dataset.root / Path(episode_path).parent / "events.json"
        event_times = []
        if events_path.exists():
            for event in json.loads(events_path.read_text()):
                if "time_interval_s" in event:
                    event_times.extend(event["time_interval_s"])
                else:
                    event_times.append(event["time_s"])
        event_times = sorted(
            t for t in event_times if time[0] <= t <= time[-1]
        )

        episodes.append({
            "name": Path(episode_path).parent.name,
            "time": time,
            "mode": mode,
            "event_times": event_times,
            "rows": rows,
            # Error magnitude per step, in units a reader can hold onto.
            "position": np.linalg.norm(errors[keep, :2], axis=1) * 1e6,  # um
            "velocity": np.linalg.norm(errors[keep, 2:], axis=1) * 1e3,  # mm/s
            "timestep": float(np.median(np.diff(time))) if len(time) > 1 else 0.0,
        })

    return episodes

def _draw_error_panel(episode, key, ax, smoothing_s):
    time = episode["time"]
    values = episode[key]

    ax.set_facecolor(SURFACE)

    # Contact mode as a background band behind the curve.
    for start, stop, code in _runs(episode["mode"], episode["rows"]):
        right = time[stop] if stop < len(time) else time[-1]
        ax.axvspan(
            time[start], right,
            facecolor=MODE_COLORS[code], alpha=0.16,
            linewidth=0, zorder=0,
        )

    # Detected transitions, marked by line
    for edge in episode["event_times"]:
        ax.axvline(
            edge, color=INK, linewidth=1.0,
            linestyle=(0, (4, 3)), alpha=0.7, zorder=3,
        )

    window = (
        round(smoothing_s / episode["timestep"])
        if episode["timestep"] > 0
        else 1
    )

    ax.plot(
        time, values,
        color=INK_MUTED, linewidth=0.5, alpha=0.25, zorder=1,
    )
    ax.plot(
        time, _rolling_rms(values, window),
        color=INK, linewidth=1.8, zorder=2,
    )

    ax.set_yscale("log")
    ax.margins(x=0)
    ax.grid(True, axis="y", color=INK, alpha=0.08, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)

def plot_episode_error_curves(
    episodes,
    figure_path,
    smoothing_s=0.01,
):
    """
    One row per validation episode: one-step prediction error against time,
    shaded by contact mode, with detected mode-transition intervals marked.
    each column shares a log y-axis across episodes so the rows are directly comparable
    """
    columns = [
        ("position", f"Position error |Δq̂ - Δq| (µm)"),
        ("velocity", f"Velocity error |Δv̂ - Δv| (mm/s)"),
    ]

    fig, axes = plt.subplots(
        len(episodes), len(columns),
        figsize=(13, 2.3 * len(episodes)),
        sharey="col",
        squeeze=False,
        layout="constrained",
    )
    fig.set_facecolor(SURFACE)

    for row, episode in enumerate(episodes):
        for column, (key, title) in enumerate(columns):
            ax = axes[row][column]
            _draw_error_panel(episode, key, ax, smoothing_s)

            if row == 0:
                ax.set_title(title, color=INK, fontsize=10, pad=8)
            if row == len(episodes) - 1:
                ax.set_xlabel("Time (s)", color=INK_MUTED, fontsize=9)

        axes[row][0].set_ylabel(
            episode["name"], color=INK, fontsize=9, labelpad=8,
        )

    for column, (key, _) in enumerate(columns):
        pooled = np.concatenate([episode[key] for episode in episodes])
        pooled = pooled[pooled > 0]
        axes[0][column].set_ylim(
            10.0 ** np.floor(np.log10(np.percentile(pooled, 1))),
            10.0 ** np.ceil(np.log10(pooled.max())),
        )

    present = sorted({int(code) for e in episodes for code in np.unique(e["mode"])})
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MODE_COLORS[code], alpha=0.16)
        for code in present
    ]
    labels = [MODE_LABELS[code] for code in present]

    handles += [
        plt.Line2D([], [], color=INK_MUTED, linewidth=0.8, alpha=0.5),
        plt.Line2D([], [], color=INK, linewidth=1.8),
        plt.Line2D([], [], color=INK, linewidth=1.0, linestyle=(0, (4, 3))),
    ]
    labels += [
        "Per-step error",
        f"Rolling RMS ({smoothing_s * 1e3:.0f} ms)",
        "Detected transition",
    ]

    fig.legend(
        handles, labels,
        loc="outside upper center",
        ncol=len(labels), frameon=False,
        fontsize=9, labelcolor=INK,
    )

    figure_path.parent.mkdir(exist_ok=True)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("Saved figure to:", figure_path)

def select_episode_signals(dataset, predicted, target, name):
    """
    One episode's raw per-dimension error lined up with the contact
    diagnostics that explain it. Everything is indexed by the rows the
    split actually retained, so the signals and the error stay aligned
    """
    predicted = predicted.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    matches = [
        episode_id
        for episode_id, episode_path in enumerate(dataset.episode_paths)
        if Path(episode_path).parent.name == name
    ]
    if not matches:
        raise ValueError(f"No episode named '{name}' in this split.")

    episode_id = matches[0]
    episode_path = dataset.episode_paths[episode_id]
    keep = dataset.episode_ids == episode_id
    rows = dataset.row_indices[keep]

    with np.load(dataset.root / episode_path) as data:
        signals = {
            "time": data["time"][rows],
            "normal_force": data["normal_force"][rows],
            "gap": data["gap"][rows] * 1e3,  # mm
        }

    events = json.loads(
        (dataset.root / Path(episode_path).parent / "events.json").read_text()
    )

    signals["name"] = name
    signals["events"] = events
    # The velocity-x target, [n], in mm/s: what actually happened over
    # [t_i, t_i+1], what the model said, and the signed difference.
    signals["vx_true"] = target[keep, 2] * 1e3
    signals["vx_predicted"] = predicted[keep, 2] * 1e3
    signals["vx_error"] = (predicted - target)[keep, 2] * 1e3
    return signals

def recontact_window_mask(record, window_s=0.005, event_type="free_to_contact"):
    """
    Samples within +/- window_s of a recontact, plus the window edges for
    the plot. a3 records the event as an instant, a2 as an interval; an
    interval is padded on both sides rather than collapsed to a point.
    """
    time = record["time"]
    mask = np.zeros(len(time), dtype=bool)
    spans = []

    for event in record["events"]:
        if event["type"] != event_type:
            continue

        if "time_interval_s" in event:
            first, last = event["time_interval_s"]
        else:
            first = last = event["time_s"]

        start, stop = first - window_s, last + window_s
        spans.append((start, stop))
        mask |= (time >= start) & (time <= stop)

    return mask, spans

def plot_recontact_windows(record, spans, figure_path, window_s=0.005):
    """
    Raw v_x error over normal force and gap on one time axis. Three
    stacked panels rather than shared axes, so no signal is read against
    another's scale; the shaded bands are the recontact windows.
    """
    panels = [
        ("vx_error", "v\u2093 error (mm/s)"),
        ("normal_force", "Normal force (N)"),
        ("gap", "Gap (mm)"),
    ]

    fig, axes = plt.subplots(
        len(panels), 1,
        figsize=(12, 8), sharex=True, squeeze=False, layout="constrained",
    )
    fig.set_facecolor(SURFACE)

    for axis, (key, label) in zip(axes[:, 0], panels):
        axis.set_facecolor(SURFACE)

        for start, stop in spans:
            axis.axvspan(start, stop, facecolor=WINDOW, alpha=0.3, linewidth=0, zorder=0)

        # Zero is a real threshold for both of these, not just a tick.
        if key in ("vx_error", "gap"):
            axis.axhline(0, color=INK_MUTED, linewidth=0.8, alpha=0.6, zorder=1)

        axis.plot(record["time"], record[key], color=INK, linewidth=0.8, zorder=2)

        axis.set_ylabel(label, color=INK, fontsize=9)
        axis.margins(x=0)
        axis.grid(True, axis="y", color=INK, alpha=0.08, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(INK_MUTED)
            axis.spines[side].set_linewidth(0.8)

    axes[-1, 0].set_xlabel("Time (s)", color=INK_MUTED, fontsize=9)
    fig.suptitle(
        f"{record['name']} \u2014 v\u2093 error against contact state "
        f"(shaded: \u00b1{window_s * 1e3:.0f} ms around recontact)",
        color=INK, fontsize=12,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("Saved figure to:", figure_path)

def _event_time(event):
    """An event's anchor time, whichever schema it was written in."""
    if "time_interval_s" in event:
        return event["time_interval_s"][0]
    return event["time_s"]

def plot_recontact_zoom(
    record,
    figure_path,
    event_index=0,
    half_width_s=0.004,
    event_type="free_to_contact",
):
    events = [e for e in record["events"] if e["type"] == event_type]
    if not events:
        raise ValueError(f"No '{event_type}' event in {record['name']}.")

    event = events[event_index]
    centre = _event_time(event)
    window = np.abs(record["time"] - centre) <= half_width_s
    offset = (record["time"][window] - centre) * 1e3  # ms

    panels = [
        ("delta", "\u0394v\u2093 over step (mm/s)"),
        ("vx_error", "v\u2093 error (mm/s)"),
        ("normal_force", "Normal force at t\u1d62 (N)"),
    ]

    fig, axes = plt.subplots(
        len(panels), 1,
        figsize=(11, 8), sharex=True, squeeze=False, layout="constrained",
    )
    fig.set_facecolor(SURFACE)

    for axis, (key, label) in zip(axes[:, 0], panels):
        axis.set_facecolor(SURFACE)
        axis.axvline(0, color=WINDOW, linewidth=1.4, zorder=1)

        if key == "delta":
            axis.plot(
                offset, record["vx_true"][window],
                color=INK, linewidth=1.0, marker="o", markersize=3.5,
                label="True", zorder=3,
            )
            axis.plot(
                offset, record["vx_predicted"][window],
                color=WINDOW, linewidth=1.0, marker="s", markersize=3.5,
                label="Predicted", zorder=2,
            )
            axis.legend(frameon=False, fontsize=9, loc="upper left")
        else:
            axis.plot(
                offset, record[key][window],
                color=INK, linewidth=1.0, marker="o", markersize=3.5, zorder=2,
            )

        if key != "normal_force":
            axis.axhline(0, color=INK_MUTED, linewidth=0.8, alpha=0.6, zorder=1)

        axis.set_ylabel(label, color=INK, fontsize=9)
        axis.grid(True, axis="y", color=INK, alpha=0.08, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(INK_MUTED)
            axis.spines[side].set_linewidth(0.8)

    axes[-1, 0].set_xlabel(
        "t\u1d62 \u2212 recontact (ms)  \u2014  t\u1d62 is the start of each "
        "prediction interval",
        color=INK_MUTED, fontsize=9,
    )
    fig.suptitle(
        f"{record['name']} \u2014 recontact {event['cycle']} at "
        f"t = {centre:.4f} s, raw samples",
        color=INK, fontsize=12,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("Saved figure to:", figure_path)

def report_transient_fit(
    record, event_index=0, half_width_s=0.004, event_type="free_to_contact"
):
    """
    Magnitude, timing and shape of the transient, separated: peak true
    against peak predicted, and the lag between where each one peaks
    """
    events = [e for e in record["events"] if e["type"] == event_type]
    event = events[event_index]
    centre = _event_time(event)
    window = np.flatnonzero(np.abs(record["time"] - centre) <= half_width_s)

    true = record["vx_true"][window]
    predicted = record["vx_predicted"][window]
    offset = (record["time"][window] - centre) * 1e3

    true_peak = int(np.argmax(np.abs(true)))
    predicted_peak = int(np.argmax(np.abs(predicted)))

    print(f"\n{record['name']} \u2014 recontact {event['cycle']}, "
          f"\u00b1{half_width_s * 1e3:.0f} ms, {len(window)} samples")
    print(f"  peak true  \u0394v_x   {true[true_peak]:+9.4f} mm/s "
          f"at t\u1d62 {offset[true_peak]:+.3f} ms")
    print(f"  peak pred  \u0394v_x   {predicted[predicted_peak]:+9.4f} mm/s "
          f"at t\u1d62 {offset[predicted_peak]:+.3f} ms")
    print(f"  magnitude captured  {abs(predicted[true_peak] / true[true_peak]):7.2%} "
          f"of the true peak, at the true peak's step")
    print(f"  timing lag          {offset[predicted_peak] - offset[true_peak]:+.3f} ms")

    return true, predicted, offset

def report_error_concentration(record, mask, window_s=0.005, label="recontact"):
    """
    What share of the samples the windows hold, against what share of the
    total squared v_x error
    """
    squared = record["vx_error"] ** 2

    sample_fraction = mask.sum() / len(mask)
    error_fraction = squared[mask].sum() / squared.sum()

    print(f"\n{record['name']} \u2014 \u00b1{window_s * 1e3:.0f} ms {label} windows")
    print(f"  samples in windows   {mask.sum():>6} / {len(mask):<6} = {sample_fraction:7.2%}")
    print(f"  squared v_x error                     = {error_fraction:7.2%}")
    print(f"  concentration                         = {error_fraction / sample_fraction:6.1f}x")

    return sample_fraction, error_fraction

def print_mode_error_table(episodes):
    print("\nMode is the label at t_i, the state each step starts in; a step so "
          "labelled\nmay still predict across a transition.")
    print(f"\n{'episode':<26}{'mode':<13}{'steps':>8}{'pos RMSE (um)':>16}{'vel RMSE (mm/s)':>18}")
    for episode in episodes:
        for code in sorted({int(c) for c in np.unique(episode["mode"])}):
            mask = episode["mode"] == code
            position = np.sqrt(np.mean(episode["position"][mask] ** 2))
            velocity = np.sqrt(np.mean(episode["velocity"][mask] ** 2))
            print(
                f"{episode['name']:<26}{MODE_LABELS[code]:<13}"
                f"{int(mask.sum()):>8}{position:>16.4g}{velocity:>18.4g}"
            )

if __name__ == "__main__":
    model, checkpoint = init_model()
    norm_stats = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in checkpoint["normalization_stats"].items()
    }
    dataset = load_dataset(norm_stats)
    x, y = dataset[0]
    print(f"Loaded dataset with {len(dataset)} samples (input {tuple(x.shape)}, target {tuple(y.shape)}) for validation.")

    predicted, target = compute_predictions(model, dataset, norm_stats)
    error_tensor = predicted - target  # [N, 4]

    # compute rmse for all batch
    sample_weighted_rmse = torch.sqrt(torch.mean(error_tensor ** 2, dim=0)) # [qx RMSE (m), qy RMSE (m), vx RMSE (m/s), vy RMSE (m/s)]
    # print("RMSE for each target dimension:", sample_weighted_rmse)

    # compute rmse for each episode
    episode_ids = torch.from_numpy(dataset.episode_ids) # [N]
    episodic_error = [] # [N_episodes, 4]
    for eid in torch.unique(episode_ids):
        mask = episode_ids == eid
        episodic_error.append(error_tensor[mask])
    episodic_rmse = []
    for error in episodic_error:
        rmse = torch.sqrt(torch.mean(error ** 2, dim=0))
        episodic_rmse.append(rmse)
    # print("RMSE for each episode:", episodic_rmse)
    episodic_rmse = torch.stack(episodic_rmse, dim=0) # [5, 4], 5 episodes, 4 target dimensions
    episode_balanced_rmse = torch.sqrt(torch.mean(episodic_rmse ** 2, dim=0)) # [4], average over episodes
    # print("Episode balanced RMSE for each target dimension:", episode_balanced_rmse)

    # RMSE comparison plot
    sample = sample_weighted_rmse.cpu().numpy()
    balanced = episode_balanced_rmse.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    width = 0.35

    panels = [
        (slice(0, 2), ["qx", "qy"], 1e6, "Position RMSE (µm)"),
        (slice(2, 4), ["vx", "vy"], 1e3, "Velocity RMSE (mm/s)"),
    ]

    for ax, (columns, labels, scale, ylabel) in zip(axes, panels):
        x = np.arange(2)

        bars_sample = ax.bar(
            x - width / 2, sample[columns] * scale,
            width, label="Sample weighted",
        )
        bars_balanced = ax.bar(
            x + width / 2, balanced[columns] * scale,
            width, label="Episode balanced",
        )

        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.bar_label(bars_sample, fmt="%.3g", padding=3)
        ax.bar_label(bars_balanced, fmt="%.3g", padding=3)
        ax.margins(y=0.25)
        ax.legend()

    plt.tight_layout()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = ANALYSIS_DIR / "rmse_comparison.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    print("Saved figure to:", figure_path)
    plt.close(fig)

    # Per-episode error curves against the contact structure.
    episodes = collect_episode_errors(dataset, error_tensor)
    plot_episode_error_curves(
        episodes,
        ANALYSIS_DIR / "episode_error_curves.png",
    )
    print_mode_error_table(episodes)

    # Is the a3 error concentrated in the moments around recontact?
    record = select_episode_signals(dataset, predicted, target, "a3_0004")
    mask, spans = recontact_window_mask(record, window_s=0.005)
    plot_recontact_windows(
        record, spans, ANALYSIS_DIR / "a3_0004_recontact.png", window_s=0.005
    )
    report_error_concentration(record, mask, window_s=0.005)

    # Zoom on one event: magnitude, timing or shape?
    plot_recontact_zoom(record, ANALYSIS_DIR / "a3_0004_recontact_zoom.png")
    report_transient_fit(record)
