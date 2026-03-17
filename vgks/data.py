from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ArrayDict = Dict[str, np.ndarray]
Trajectory = Dict[str, np.ndarray]
PathLike = Union[str, Path]


def _normalize_array_dict(data: Dict[str, object]) -> ArrayDict:
    return {
        key: None if value is None else np.asarray(value, dtype=np.float32)
        for key, value in data.items()
    }


def flatten_trajectory_list(paths: List[Trajectory]) -> ArrayDict:
    if not paths:
        raise ValueError("Trajectory list is empty")
    keys = paths[0].keys()
    flattened = {}
    for key in keys:
        values = [np.asarray(path[key], dtype=np.float32) for path in paths if key in path and path[key] is not None]
        flattened[key] = np.concatenate(values, axis=0) if values else None
    return flattened


def replay_to_trajectory_list(data: ArrayDict) -> List[Trajectory]:
    normalized = _normalize_array_dict(data)
    terminals = normalized.get("terminals")
    num_steps = int(normalized["observations"].shape[0])
    if terminals is None:
        terminals = np.zeros(num_steps, dtype=np.float32)
        terminals[-1] = 1.0

    paths: List[Trajectory] = []
    start = 0
    for idx in range(num_steps):
        terminal = bool(terminals[idx]) or idx == num_steps - 1
        if terminal:
            path = {}
            for key, value in normalized.items():
                if value is None:
                    continue
                path[key] = np.asarray(value[start : idx + 1], dtype=np.float32)
            paths.append(path)
            start = idx + 1
    return paths


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
        return _normalize_array_dict(source)

    source = Path(source)
    if source.is_dir():
        pkl_path = source / "dataset.pkl"
        if pkl_path.exists():
            with pkl_path.open("rb") as handle:
                loaded = pickle.load(handle)
            if isinstance(loaded, list):
                return flatten_trajectory_list(loaded)
            return _normalize_array_dict(loaded)
        pkl_candidates = sorted(source.glob("*.pkl"))
        if pkl_candidates:
            return load_offline_dataset(pkl_candidates[0])
        raise ValueError(f"Dataset directory '{source}' does not contain dataset.pkl")

    if source.suffix == ".npz":
        with np.load(source) as data:
            return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}

    if source.suffix == ".pkl":
        with source.open("rb") as handle:
            loaded = pickle.load(handle)
        if isinstance(loaded, list):
            return flatten_trajectory_list(loaded)
        return _normalize_array_dict(loaded)

    if source.suffix == ".npy":
        loaded = np.load(source, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.dtype == object:
            return flatten_trajectory_list(list(loaded))
        raise ValueError(f"Unsupported .npy dataset structure in '{source}'")

    if source.suffix == ".pt":
        loaded = torch.load(source, map_location="cpu")
        return {key: np.asarray(value, dtype=np.float32) for key, value in loaded.items()}

    raise ValueError(
        f"Unsupported dataset source '{source}'. Use a dict, directory with dataset.pkl, .pkl, .npz, or .pt file."
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


def save_trajectory_cache(cache_dir: PathLike, data: ArrayDict) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_array_dict(data)
    with (cache_dir / "dataset.pkl").open("wb") as handle:
        pickle.dump(normalized, handle)

    for key, value in normalized.items():
        if value is None:
            continue
        np.save(cache_dir / f"{key}.npy", value)
    return cache_dir


def save_trajectory_paths(output_dir: PathLike, env_name: str, paths: List[Trajectory]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_paths = [_normalize_array_dict(path) for path in paths]
    prefix = output_dir / env_name
    with prefix.with_suffix(".pkl").open("wb") as handle:
        pickle.dump(normalized_paths, handle)
    np.save(prefix.with_suffix(".npy"), np.array(normalized_paths, dtype=object), allow_pickle=True)
    return prefix


def build_dataloader(
    dataset: OfflineReplayDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
