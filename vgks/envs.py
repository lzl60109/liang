from __future__ import annotations

from typing import Dict


def resolve_env_name(
    env_name: str = None, task: str = None, dataset_name: str = None, version: str = "v2"
) -> str:
    if env_name:
        return env_name
    if task and dataset_name:
        return f"{task}-{dataset_name}-{version}"
    return None


def make_env(env_name: str):
    try:
        import gym  # type: ignore
    except ImportError as exc:
        raise ImportError("gym is required to create D4RL environments") from exc

    try:
        import d4rl  # type: ignore  # noqa: F401
    except ImportError:
        pass

    return gym.make(env_name)


def infer_env_dims(env) -> Dict[str, int]:
    return {
        "state_dim": int(env.observation_space.shape[0]),
        "action_dim": int(env.action_space.shape[0]),
    }
