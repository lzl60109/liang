from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ArrayDict = Dict[str, np.ndarray]
PathLike = Union[str, Path]


class OfflineReplayDataset(Dataset):
    def __init__(self, data: ArrayDict) -> None:
        required = {"observations", "actions", "next_observations"}
        missing = required.difference(data)
        if missing:
            raise KeyError(f"Dataset is missing required keys: {sorted(missing)}")
        self.data = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}

    def __len__(self) -> int:
        return int(self.data["observations"].shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {}
        for key, value in self.data.items():
            if value is None:
                continue
            item[key] = torch.tensor(value[index], dtype=torch.float32)
        return item


def load_offline_dataset(source: Union[PathLike, ArrayDict]) -> ArrayDict:
    if isinstance(source, dict):
        return {key: np.asarray(value, dtype=np.float32) for key, value in source.items()}

    source = Path(source)
    if source.suffix == ".npz":
        with np.load(source) as data:
            return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}

    if source.suffix == ".pt":
        loaded = torch.load(source, map_location="cpu")
        return {key: np.asarray(value, dtype=np.float32) for key, value in loaded.items()}

    raise ValueError(
        f"Unsupported dataset source '{source}'. Use a dict, .npz, or .pt file."
    )


def load_d4rl_dataset(env_name: str) -> ArrayDict:
    try:
        import d4rl  # type: ignore
        import gym  # type: ignore
    except ImportError as exc:
        raise ImportError("d4rl and gym are required to load a D4RL dataset") from exc

    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    return {
        "observations": np.asarray(dataset["observations"], dtype=np.float32),
        "actions": np.asarray(dataset["actions"], dtype=np.float32),
        "next_observations": np.asarray(dataset["next_observations"], dtype=np.float32),
        "rewards": np.asarray(dataset["rewards"], dtype=np.float32)
        if "rewards" in dataset
        else None,
        "terminals": np.asarray(dataset["terminals"], dtype=np.float32)
        if "terminals" in dataset
        else None,
    }


def build_dataloader(
    dataset: OfflineReplayDataset, batch_size: int = 256, shuffle: bool = True
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
