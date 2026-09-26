import argparse
import contextlib
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

ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"

TRAINING_RUNS = Path(__file__).resolve().parent / "training_runs"

DEFAULT_SEED = 42

# validation results stay where they were and test results go to test_split/
SPLITS = ("validation", "test")
DEFAULT_SPLIT = "validation"

RUN_ROOT = Path(__file__).resolve().parent / "run_001"

FAMILY_ORDER = ("a3", "a3_contact_control", "a3_free_control")
FAMILY_LABELS = {
    "a3": "separation / recontact",
    "a3_contact_control": "sustained-contact control",
    "a3_free_control": "free-flight control",
}

def checkpoint_path(seed, variant):
    """
    The one best.pt for this seed and variant, from training_runs/seed<N>/
    """
    seed_dir = TRAINING_RUNS / f"seed{seed}"
    paths = sorted(seed_dir.glob(f"*_{variant}/best.pt"))
    if len(paths) != 1:
        raise SystemExit(
            f"Expected one {variant} run in {seed_dir}, found {len(paths)}."
        )
    return paths[0]


def split_output_dir(split=DEFAULT_SPLIT):
    """analysis/ for the validation split, analysis/test_split/ for test"""
    if split not in SPLITS:
        raise SystemExit(f"Unknown split {split!r}; expected one of {SPLITS}.")
    return ANALYSIS_DIR if split == "validation" else ANALYSIS_DIR / "test_split"


def seed_output_dir(seed, split=DEFAULT_SPLIT):
    """
    <split dir>/seed<N>/, where every result for this seed is written. Both
    runs are checked first, so a mistyped seed leaves no empty directory.
    """
    for variant in ("original", "modified"):
        checkpoint_path(seed, variant)
    return split_output_dir(split) / f"seed{seed}"


def add_split_argument(parser):
    parser.add_argument("--split", choices=SPLITS, default=DEFAULT_SPLIT,
                        help="Episodes to evaluate on. test writes to "
                             "analysis/test_split/.")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def log_to(path):
    """Print as usual and keep a copy of everything printed in path"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as log, contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        yield
    print("Saved log to:", path)


def init_model(seed=DEFAULT_SEED):
    loaded = []
    for variant in ("original", "modified"):
        path = checkpoint_path(seed, variant)
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint["config"]["seed"] != seed:
            raise SystemExit(
                f"{path} was trained with seed {checkpoint['config']['seed']}, "
                f"not {seed}."
            )

        model = dynamicsMLP()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("Model loaded from:", path)

        loaded += [model, checkpoint]

    model_0, checkpoint_0, model_0_9, checkpoint_0_9 = loaded
    return model_0, checkpoint_0, model_0_9, checkpoint_0_9

def load_dataset(norm_stats, is_modified=False, split=DEFAULT_SPLIT):
    variant = "modified" if is_modified else "original"
    root = Path(__file__).resolve().parent / "run_001" / variant

    dataset = E1Dataset(
        roots = root,
        split = split,
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

def available_seeds():
    """Every seed with a training_runs/seed<N>/ directory, in numeric order"""
    return sorted(
        int(path.name[len("seed"):])
        for path in TRAINING_RUNS.glob("seed*")
        if path.is_dir() and path.name[len("seed"):].isdigit()
    )


def load_bundles(seed=DEFAULT_SEED, split=DEFAULT_SPLIT):
    print(f"Split: {split}")
    model_0, checkpoint_0, model_0_9, checkpoint_0_9 = init_model(seed)
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
        dataset = load_dataset(stats, is_modified, split)
        predicted, target = compute_predictions(model, dataset, stats)
        bundles[condition] = (dataset, predicted, target)
    return bundles


def episode_families(run_root=RUN_ROOT):
    """
    Identifier -> family, taken from the pair manifest
    """
    pairs = json.loads((run_root / "pair_manifest.json").read_text())["pairs"]
    return {pair["id"]: pair["family"] for pair in pairs}


def episode_vx_rmse(dataset, error):
    """
    v_x RMSE per episode, in mm/s, keyed by the episode's saved identifier
    """
    vx_error = error.detach().cpu().numpy()[:, 2] * 1e3  # mm/s

    rmse = {}
    for episode_id, episode_path in enumerate(dataset.episode_paths):
        keep = dataset.episode_ids == episode_id
        identifier = Path(episode_path).parent.name
        rmse[identifier] = float(np.sqrt(np.mean(vx_error[keep] ** 2)))

    return rmse


def print_family_vx_comparison(
    rmse_original,
    rmse_modified,
    families=None,
    label_original="original",
    label_modified="modified",
):
    """One table per family: original against modified v_x RMSE, matched"""
    families = episode_families() if families is None else families

    matched = set(rmse_original) & set(rmse_modified)
    unmatched = set(rmse_original) ^ set(rmse_modified)
    if unmatched:
        print(f"  WARNING unmatched episodes, omitted: {sorted(unmatched)}")

    unknown = sorted(i for i in matched if i not in families)
    if unknown:
        print(f"  WARNING no family recorded, omitted: {unknown}")

    header = (f"    {'episode':<28}{label_original:>14}{label_modified:>14}"
              f"{'modified - original':>22}{'change':>10}")

    for family in FAMILY_ORDER:
        members = sorted(i for i in matched if families.get(i) == family)
        if not members:
            continue

        print(f"\n  [{family}]  {FAMILY_LABELS.get(family, '')}"
              f"   v_x RMSE in mm/s")
        print(header)
        for identifier in members:
            before = rmse_original[identifier]
            after = rmse_modified[identifier]
            change = (after - before) / before if before else float("nan")
            print(f"    {identifier:<28}{before:>14.6f}{after:>14.6f}"
                  f"{after - before:>+22.6f}{change:>+10.1%}")

        # Episode-balanced, so a long episode cannot dominate the family
        before = float(np.sqrt(np.mean([rmse_original[i] ** 2 for i in members])))
        after = float(np.sqrt(np.mean([rmse_modified[i] ** 2 for i in members])))
        change = (after - before) / before if before else float("nan")
        print(f"    {'family (episode-balanced)':<28}{before:>14.6f}{after:>14.6f}"
              f"{after - before:>+22.6f}{change:>+10.1%}")


def main(seed, split=DEFAULT_SPLIT):
    print(f"Split: {split}")
    model_0, checkpoint_0, model_0_9, checkpoint_0_9 = init_model(seed)
    norm_state_0 = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in checkpoint_0["normalization_stats"].items()
    }
    norm_state_0_9 = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in checkpoint_0_9["normalization_stats"].items()
    }

    dataset_0 = load_dataset(norm_state_0, split=split)
    dataset_0_9 = load_dataset(norm_state_0_9, True, split)

    predicted_0, target_0 = compute_predictions(model_0, dataset_0, norm_state_0)
    predicted_0_9, target_0_9 = compute_predictions(model_0_9, dataset_0_9, norm_state_0_9)

    error_0 = predicted_0 - target_0
    error_0_9 = predicted_0_9 - target_0_9

    sample_weighted_rmse_0 = torch.sqrt(torch.mean(error_0 ** 2, dim=0)) # [qx RMSE (m), qy RMSE (m), vx RMSE (m/s), vy RMSE (m/s)]
    sample_weighted_rmse_0_9 = torch.sqrt(torch.mean(error_0_9 ** 2, dim=0))

    # print RMSE for different models
    print("#"*80)
    print("RMSE for each target dimension (original model):", sample_weighted_rmse_0)
    print("RMSE for each target dimension (modified model):", sample_weighted_rmse_0_9)
    print("#"*80)

    # Per-episode v_x, matched by identifier and reported per family
    print("\nPer-episode v_x RMSE, matched by episode identifier")
    print_family_vx_comparison(
        episode_vx_rmse(dataset_0, error_0),
        episode_vx_rmse(dataset_0_9, error_0_9),
        label_original=checkpoint_0["config"]["variant"],
        label_modified=checkpoint_0_9["config"]["variant"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E1A: original against modified, for one training seed."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Which training_runs/seed<N>/ to evaluate.")
    add_split_argument(parser)
    args = parser.parse_args()

    with log_to(seed_output_dir(args.seed, args.split) / "diagnose.txt"):
        main(args.seed, args.split)
