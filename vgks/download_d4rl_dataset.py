from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vgks.data import load_d4rl_dataset, save_trajectory_cache, save_trajectory_paths
from vgks.envs import resolve_env_name


def split_d4rl_trajectories(env_name: str):
    try:
        import d4rl  # type: ignore
        import gym  # type: ignore
    except ImportError as exc:
        raise ImportError("d4rl and gym are required to download D4RL trajectories") from exc

    env = gym.make(env_name)
    q_dataset = d4rl.qlearning_dataset(env)
    raw_dataset = env.get_dataset()
    use_timeouts = "timeouts" in raw_dataset

    paths = []
    current = defaultdict(list)
    num_steps = q_dataset["observations"].shape[0]

    for idx in range(num_steps):
        for key in ["observations", "next_observations", "actions", "rewards", "terminals"]:
            current[key].append(q_dataset[key][idx])

        done = bool(q_dataset["terminals"][idx])
        timeout = bool(raw_dataset["timeouts"][idx]) if use_timeouts and idx < len(raw_dataset["timeouts"]) else False
        discontinuity = False
        if idx + 1 < num_steps:
            discontinuity = (
                np.linalg.norm(q_dataset["observations"][idx + 1] - q_dataset["next_observations"][idx]) > 1e-6
            )

        if done or timeout or discontinuity or idx == num_steps - 1:
            paths.append(
                {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in current.items()
                }
            )
            current = defaultdict(list)

    return paths


def download_and_cache_dataset(
    *,
    env_name: str = None,
    task: str = None,
    dataset_name: str = None,
    output_dir: Path,
) -> Path:
    resolved_env_name = resolve_env_name(env_name=env_name, task=task, dataset_name=dataset_name)
    if resolved_env_name is None:
        raise ValueError("Provide either env_name or task + dataset_name")
    output_dir = Path(output_dir)

    paths = split_d4rl_trajectories(resolved_env_name)
    prefix = save_trajectory_paths(output_dir, resolved_env_name, paths)

    # Keep the old flat cache directory for backward compatibility with earlier configs.
    data = load_d4rl_dataset(resolved_env_name)
    cache_dir = output_dir / resolved_env_name
    save_trajectory_cache(cache_dir, data)

    (output_dir / f"{resolved_env_name}.json").write_text(
        json.dumps({"env_name": resolved_env_name, "num_trajectories": len(paths)}, indent=2),
        encoding="utf-8",
    )
    (cache_dir / "meta.json").write_text(
        json.dumps({"env_name": resolved_env_name, "num_trajectories": len(paths)}, indent=2),
        encoding="utf-8",
    )
    return prefix


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description="Download and cache D4RL dataset")
    parser.add_argument("--env-name", dest="env_name", type=str, default=None)
    parser.add_argument("--task", dest="task", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    args = parser.parse_args()

    cache_dir = download_and_cache_dataset(
        env_name=args.env_name,
        task=args.task,
        dataset_name=args.dataset_name,
        output_dir=Path(args.output_dir),
    )
    print(cache_dir)


if __name__ == "__main__":
    main()
