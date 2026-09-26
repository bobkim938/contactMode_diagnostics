import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from dataset import E1Dataset
from model import dynamicsMLP

"""
python model/train.py 
python model/train.py --experiment E1A  # E1A, original (default solimp[0]=0.9)
python model/train.py --experiment E1A --variant modified
"""


# Where each experiment keeps its data. E1 has a single dataset root;
# E1A carries the paired contact configurations from the A3 sweep, so it
# needs a variant as well.
EXPERIMENTS = {
    "E1": {"variants": ()},
    "E1A": {"variants": ("original", "modified")},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the dynamics MLP on one experiment's dataset."
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="E1",
        help="Which experiment to train on. Runs are saved under it.",
    )
    parser.add_argument(
        "--variant",
        choices=("original", "modified"),
        default=None,
        help="Contact configuration. E1A only; defaults to 'original'.",
    )
    parser.add_argument(
        "--dataset-name",
        default="run_001",
        help="Dataset directory inside the experiment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for weight init and training shuffle order.",
    )
    return parser.parse_args()


def resolve_experiment(args):
    project_root = Path(__file__).resolve().parent.parent
    experiment_dir = project_root / args.experiment
    variants = EXPERIMENTS[args.experiment]["variants"]

    variant = args.variant
    if variants:
        variant = variant or variants[0]
    elif variant is not None:
        raise SystemExit(
            f"{args.experiment} has no variants; drop --variant."
        )

    dataset_root = experiment_dir / args.dataset_name
    if variant:
        dataset_root = dataset_root / variant

    if not (dataset_root / "accepted_splits.json").is_file():
        raise SystemExit(f"No accepted_splits.json under {dataset_root}.")

    # The variant goes in the run name so E1A's two configurations stay
    # apart in one training_runs directory.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if variant:
        run_name = f"{run_name}_{variant}"

    return dataset_root, experiment_dir / "training_runs" / f"seed{args.seed}" / run_name, variant


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Training loss became non-finite.")

        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Training dataset is empty.")

    return total_loss / total_examples


def evaluate(model, loader, device):
    # Evaluation order must match the dataset's episode metadata.
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Evaluation requires shuffle=False.")

    if loader.drop_last:
        raise ValueError("Evaluation requires drop_last=False.")

    model.eval()
    loss_parts = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            # One normalized MSE value per transition.
            per_example_loss = (
                (predictions - targets).square().mean(dim=1)
            )

            loss_parts.append(per_example_loss.cpu().numpy())

    if not loss_parts:
        raise ValueError("Evaluation dataset is empty.")

    example_losses = np.concatenate(loss_parts).astype(np.float64)

    if not np.isfinite(example_losses).all():
        raise RuntimeError("Evaluation loss became non-finite.")

    dataset = loader.dataset
    episode_ids = dataset.episode_ids

    if len(example_losses) != len(episode_ids):
        raise ValueError(
            "Prediction errors and episode metadata have different lengths."
        )

    per_episode = {}

    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        episode_path = dataset.episode_paths[int(episode_id)]

        per_episode[episode_path] = float(
            example_losses[mask].mean()
        )

    return {
        "sample_mean": float(example_losses.mean()),
        "episode_mean": float(np.mean(list(per_episode.values()))),
        "per_episode": per_episode,
    }


def main():
    args = parse_args()
    dataset_root, output_dir, variant = resolve_experiment(args)

    print("Experiment:", args.experiment, f"({variant})" if variant else "")
    print("Dataset:", dataset_root)

    # 1. Configuration
    seed = args.seed
    batch_size = 1024
    learning_rate = 1e-3
    max_epochs = 200
    patience = 20

    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)

    # 2. Datasets and loaders
    # An absolute root is taken as given by E1Dataset, so the data can
    # live outside model/.
    train_dataset = E1Dataset(
        roots=dataset_root,
        split="train",
    )

    validation_dataset = E1Dataset(
        roots=dataset_root,
        split="validation",
        stats=train_dataset.stats,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    # 3. Model, loss, and optimizer
    model = dynamicsMLP().to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # 4. Output directory, under the experiment this run belongs to
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Saving results to:", output_dir)

    # 5. Tracking variables
    best_validation_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    stopped_early = False

    history = {
        "epoch": [],
        "train_loss": [],
        "validation_sample_loss": [],
        "validation_episode_loss": [],
        "validation_per_episode": [],
    }

    config = {
        "experiment": args.experiment,
        "variant": variant,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "patience": patience,
        "dataset_root": str(train_dataset.root.resolve()),
        "selection_metric": "validation_episode_mean",
        "model_class": type(model).__name__,
        "model_description": str(model),
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False)
    )

    # 6. Training loop
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = evaluate(
            model=model,
            loader=validation_loader,
            device=device,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_sample_loss"].append(
            validation_metrics["sample_mean"]
        )
        history["validation_episode_loss"].append(
            validation_metrics["episode_mean"]
        )
        history["validation_per_episode"].append(
            validation_metrics["per_episode"]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.6f} | "
            f"Val samples: {validation_metrics['sample_mean']:.6f} | "
            f"Val episodes: {validation_metrics['episode_mean']:.6f}"
        )

        current_score = validation_metrics["episode_mean"]

        # Save every strict improvement in the selection metric
        if current_score < best_validation_loss:
            best_validation_loss = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_metrics": validation_metrics,

                # Preserve training normalization in float64.
                "normalization_stats": {
                    name: torch.tensor(
                        values,
                        dtype=torch.float64,
                        device="cpu",
                    )
                    for name, values in train_dataset.stats.items()
                },

                "config": config,
                "episode_splits": train_dataset.splits,
            }

            torch.save(checkpoint, output_dir / "best.pt")

            print(
                f"  Saved best checkpoint: epoch {epoch}, "
                f"score {current_score:.6f}"
            )

        else:
            epochs_without_improvement += 1

            print(
                f"  No improvement: "
                f"{epochs_without_improvement}/{patience}"
            )

        # Save history before checking the stopping condition
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False)
        )

        if epochs_without_improvement >= patience:
            stopped_early = True

            print(
                f"\nEarly stopping at epoch {epoch}: "
                f"no validation improvement for {patience} epochs."
            )
            break

    print(
        f"\nBest epoch: {best_epoch} | "
        f"Best validation episode loss: "
        f"{best_validation_loss:.6f}"
    )

    # 7. Reload and verify the best checkpoint
    checkpoint = torch.load(
        output_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    restored_model = dynamicsMLP().to(device)
    restored_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # The existing loader already uses the saved training statistics.
    restored_metrics = evaluate(
        model=restored_model,
        loader=validation_loader,
        device=device,
    )

    saved_score = checkpoint["validation_metrics"]["episode_mean"]
    restored_score = restored_metrics["episode_mean"]

    print(f"Saved checkpoint score:    {saved_score:.8f}")
    print(f"Restored checkpoint score: {restored_score:.8f}")

    if not np.isclose(
        restored_score,
        saved_score,
        rtol=1e-5,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Restored model did not reproduce the saved validation score."
        )

    print("Checkpoint verification passed.")

    print("\nBest model: validation loss per episode")
    for episode_path, episode_loss in restored_metrics["per_episode"].items():
        print(f"  {episode_path}: {episode_loss:.6f}")

    # 8. Save the final run summary
    summary = {
        "epochs_completed": len(history["epoch"]),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation_episode_loss": best_validation_loss,
        "restored_validation_metrics": restored_metrics,
        "checkpoint_verified": True,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )

    print("\nResults directory:", output_dir)
    print("Checkpoint:", output_dir / "best.pt")


if __name__ == "__main__":
    main()