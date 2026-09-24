import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class E1Dataset(Dataset):
    def __init__(self, roots=None, split="train", stats=None):
        """
        roots: Directory containing accepted_splits.json.
        split: "train", "validation", or "test".
        stats: Training normalization statistics.
               Required for validation and test.
        """
        if roots is None:
            root = Path(__file__).resolve().parent / "run_001"
        else:
            # An absolute roots is used as given; a relative one still
            # resolves against this file, as before.
            root = Path(__file__).resolve().parent / roots

        self.root = Path(root)
        self.split = split

        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {split}")

        if split != "train" and stats is None:
            raise ValueError(
                "Validation/test must receive training statistics."
            )

        self.splits = json.loads(
            (self.root / "accepted_splits.json").read_text()
        )
        self.episode_paths = self.splits[split]

        input_parts = []
        target_parts = []

        # Preserve each example's origin for later E1 analysis.
        episode_id_parts = []
        row_index_parts = []

        for episode_id, episode_path in enumerate(self.episode_paths):
            with np.load(self.root / episode_path) as data:
                keep = (
                    data["analysis_mask"]
                    & data["valid_transition"]
                )
                filtered = np.flatnonzero(keep)

                episode_inputs = np.concatenate(
                    [
                        data["qpos"][filtered],
                        data["qvel"][filtered],
                        data["ctrl"][filtered],
                    ],
                    axis=1,
                )

                qpos_delta = (
                    data["qpos_next"][filtered]
                    - data["qpos"][filtered]
                )
                qvel_delta = (
                    data["qvel_next"][filtered]
                    - data["qvel"][filtered]
                )

                episode_targets = np.concatenate(
                    [qpos_delta, qvel_delta],
                    axis=1,
                )

                input_parts.append(episode_inputs)
                target_parts.append(episode_targets)

                episode_id_parts.append(
                    np.full(len(filtered), episode_id, dtype=np.int64)
                )
                row_index_parts.append(filtered)

        if not input_parts:
            raise ValueError(f"No episodes found for split '{split}'.")

        # These operations happen AFTER the episode loop.
        inputs = np.concatenate(input_parts, axis=0)
        targets = np.concatenate(target_parts, axis=0)

        if len(inputs) == 0:
            raise ValueError(f"No eligible examples in split '{split}'.")

        self.episode_ids = np.concatenate(episode_id_parts)
        self.row_indices = np.concatenate(row_index_parts)

        # Only the training dataset may calculate new statistics
        if stats is None:
            input_std = inputs.std(axis=0)
            target_std = targets.std(axis=0)

            stats = {
                "input_mean": inputs.mean(axis=0),
                "input_scale": np.where(input_std > 0, input_std, 1.0),
                "target_mean": targets.mean(axis=0),
                "target_scale": np.where(target_std > 0, target_std, 1.0),
            }

        self.stats = stats

        inputs = (
            inputs - self.stats["input_mean"]
        ) / self.stats["input_scale"]

        targets = (
            targets - self.stats["target_mean"]
        ) / self.stats["target_scale"]

        # convert to float32 after 
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

if __name__ == "__main__":
    train_dataset = E1Dataset(roots="run_001", split="train")

    validation_dataset = E1Dataset(
        roots="run_001",
        split="validation",
        stats=train_dataset.stats,
    )

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    inputs, targets = train_dataset[0]

    print("Input shape:", inputs.shape)    # torch.Size([6])
    print("Target shape:", targets.shape)  # torch.Size([4])
    print("Input dtype:", inputs.dtype)    # torch.float32

    assert torch.isfinite(train_dataset.inputs).all()
    assert torch.isfinite(train_dataset.targets).all()

    print("Training input means:", train_dataset.inputs.mean(dim=0))
    print("Training target means:", train_dataset.targets.mean(dim=0))

    batch_size = 1024

    generator = torch.Generator()
    generator.manual_seed(2)

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

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    batch_inputs, batch_targets = next(iter(train_loader))

    print("Batch input shape:", batch_inputs.shape)
    print("Batch target shape:", batch_targets.shape)

    assert batch_inputs.shape[1] == 6
    assert batch_targets.shape[1] == 4
    assert batch_inputs.shape[0] == batch_targets.shape[0]
    assert torch.isfinite(batch_inputs).all()
    assert torch.isfinite(batch_targets).all()
