from __future__ import annotations

import json
from pathlib import Path

from vgks.data import load_d4rl_dataset, save_trajectory_cache
from vgks.envs import resolve_env_name


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
    data = load_d4rl_dataset(resolved_env_name)
    cache_dir = Path(output_dir) / resolved_env_name
    save_trajectory_cache(cache_dir, data)
    (cache_dir / "meta.json").write_text(
        json.dumps({"env_name": resolved_env_name}, indent=2), encoding="utf-8"
    )
    return cache_dir


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
