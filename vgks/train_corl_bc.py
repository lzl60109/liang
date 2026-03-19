from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vgks.data import load_d4rl_dataset, load_offline_dataset
from vgks.envs import infer_env_dims, make_env, resolve_env_name
from vgks.eval import evaluate_policy
from vgks.experiment_logging import ExperimentLogger
from vgks.train_bc import ToyEvalEnv


TensorBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(axis=0)
    std = states.std(axis=0) + eps
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((states - mean) / std).astype(np.float32)


def _ensure_replay_fields(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normalized = {key: np.asarray(value, dtype=np.float32) for key, value in data.items() if value is not None}
    count = int(normalized["observations"].shape[0])
    if "rewards" not in normalized:
        normalized["rewards"] = np.zeros(count, dtype=np.float32)
    if "terminals" not in normalized:
        normalized["terminals"] = np.zeros(count, dtype=np.float32)
    return normalized


def _concatenate_optional(
    raw_data: Dict[str, np.ndarray],
    aug_data: Dict[str, np.ndarray],
    indices: np.ndarray,
    key: str,
) -> np.ndarray:
    raw_value = np.asarray(raw_data.get(key), dtype=np.float32)
    if key in aug_data and aug_data[key] is not None:
        aug_value = np.asarray(aug_data[key], dtype=np.float32)[indices]
    else:
        if raw_value.ndim == 1:
            aug_value = np.zeros(indices.shape[0], dtype=np.float32)
        else:
            aug_value = np.zeros((indices.shape[0],) + raw_value.shape[1:], dtype=np.float32)
    return np.concatenate([raw_value, aug_value], axis=0).astype(np.float32)


def build_corl_bc_training_data(
    *,
    dataset_path: Optional[Path] = None,
    raw_dataset_path: Optional[Path] = None,
    aug_dataset_path: Optional[Path] = None,
    env_name: Optional[str] = None,
    mix_aug_ratio: float = 0.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    if dataset_path is not None:
        return _ensure_replay_fields(load_offline_dataset(dataset_path))

    if raw_dataset_path is None:
        return _ensure_replay_fields(load_d4rl_dataset(env_name))

    raw_data = _ensure_replay_fields(load_offline_dataset(raw_dataset_path))
    if aug_dataset_path is None or mix_aug_ratio <= 0.0:
        return raw_data

    aug_data = _ensure_replay_fields(load_offline_dataset(aug_dataset_path))
    raw_size = int(raw_data["observations"].shape[0])
    aug_size = int(aug_data["observations"].shape[0])
    take = min(aug_size, int(round(raw_size * mix_aug_ratio)))
    if take <= 0:
        return raw_data

    rng = np.random.default_rng(seed)
    indices = rng.choice(aug_size, size=take, replace=False)
    mixed = {
        "observations": np.concatenate(
            [np.asarray(raw_data["observations"], dtype=np.float32), np.asarray(aug_data["observations"], dtype=np.float32)[indices]],
            axis=0,
        ).astype(np.float32),
        "actions": np.concatenate(
            [np.asarray(raw_data["actions"], dtype=np.float32), np.asarray(aug_data["actions"], dtype=np.float32)[indices]],
            axis=0,
        ).astype(np.float32),
        "next_observations": np.concatenate(
            [
                np.asarray(raw_data["next_observations"], dtype=np.float32),
                np.asarray(aug_data["next_observations"], dtype=np.float32)[indices],
            ],
            axis=0,
        ).astype(np.float32),
        "rewards": _concatenate_optional(raw_data, aug_data, indices, "rewards"),
        "terminals": _concatenate_optional(raw_data, aug_data, indices, "terminals"),
    }
    return mixed


def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
) -> Dict[str, np.ndarray]:
    if frac >= 1.0:
        return dataset

    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0.0
    reward_scale = 1.0

    for idx, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        cur_return += reward_scale * float(reward)
        cur_ids.append(idx)
        reward_scale *= discount
        if float(done) == 1.0 or len(cur_ids) == max_episode_steps or idx == len(dataset["rewards"]) - 1:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0.0
            reward_scale = 1.0

    sort_ord = np.argsort(np.asarray(returns, dtype=np.float32))[::-1]
    top_count = max(1, int(round(frac * len(sort_ord))))
    top_trajs = sort_ord[:top_count]
    order = np.concatenate([np.asarray(ids_by_trajectories[i], dtype=np.int64) for i in top_trajs], axis=0)

    return {key: np.asarray(value, dtype=np.float32)[order] for key, value in dataset.items()}


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str) -> None:
        self._buffer_size = int(buffer_size)
        self._size = 0
        self._pointer = 0
        self._device = device
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def load_dataset(self, data: Dict[str, np.ndarray]) -> None:
        if self._size != 0:
            raise ValueError("ReplayBuffer already contains data")
        count = int(data["observations"].shape[0])
        if count > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset")

        self._states[:count] = torch.as_tensor(data["observations"], dtype=torch.float32, device=self._device)
        self._actions[:count] = torch.as_tensor(data["actions"], dtype=torch.float32, device=self._device)
        self._rewards[:count] = torch.as_tensor(data["rewards"], dtype=torch.float32, device=self._device).unsqueeze(-1)
        self._next_states[:count] = torch.as_tensor(data["next_observations"], dtype=torch.float32, device=self._device)
        self._dones[:count] = torch.as_tensor(data["terminals"], dtype=torch.float32, device=self._device).unsqueeze(-1)
        self._size = count
        self._pointer = count

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._pointer, size=batch_size)
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float) -> None:
        super().__init__()
        self.max_action = float(max_action)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class NormalizedActorAdapter(nn.Module):
    def __init__(self, actor: Actor, state_mean: np.ndarray, state_std: np.ndarray) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations.float() - self.state_mean) / self.state_std
        return self.actor(normalized)


class BCTrainer:
    def __init__(self, actor: Actor, actor_optimizer: torch.optim.Optimizer) -> None:
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.total_it = 0

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, _, _, _ = batch
        predicted = self.actor(states)
        loss = F.mse_loss(predicted, actions)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return {"actor_loss": float(loss.detach().cpu().item())}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }


def _make_eval_env(env_name: Optional[str], state_dim: int, action_dim: int):
    if env_name is None:
        return ToyEvalEnv(state_dim, action_dim)
    return make_env(env_name)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_train(step: int, total_steps: int, metrics: Dict[str, float]) -> str:
    return f"[Train][CORL-BC] step={step}/{total_steps} actor_loss={metrics['actor_loss']:.4f}"


def _format_eval(step: int, total_steps: int, metrics: Dict[str, float]) -> str:
    return (
        f"[Eval][CORL-BC] step={step}/{total_steps} return={metrics['raw_return']:.4f} "
        f"normalized_score={metrics['normalized_score']:.4f} episodes={metrics['episodes']:.4f}"
    )


def run_corl_bc_training(
    *,
    dataset_path: Optional[Path],
    raw_dataset_path: Optional[Path] = None,
    aug_dataset_path: Optional[Path] = None,
    mix_aug_ratio: float = 0.0,
    env_name: Optional[str],
    state_dim: int,
    action_dim: int,
    batch_size: int,
    max_timesteps: int,
    eval_freq: int,
    log_every: int,
    seed: int,
    device: str,
    save_dir: Path,
    use_wandb: bool,
    wandb_project: str,
    wandb_group: str,
    wandb_name: str,
    eval_episodes: int = 10,
    frac: float = 1.0,
    discount: float = 0.99,
    buffer_size: int = 2_000_000,
    normalize: bool = True,
    max_episode_steps: int = 1000,
) -> Dict[str, Dict[str, float]]:
    _set_seed(seed)

    data = build_corl_bc_training_data(
        dataset_path=dataset_path,
        raw_dataset_path=raw_dataset_path,
        aug_dataset_path=aug_dataset_path,
        env_name=env_name,
        mix_aug_ratio=mix_aug_ratio,
        seed=seed,
    )
    data = keep_best_trajectories(data, frac=frac, discount=discount, max_episode_steps=max_episode_steps)

    if normalize:
        state_mean, state_std = compute_mean_std(data["observations"])
    else:
        state_mean = np.zeros(state_dim, dtype=np.float32)
        state_std = np.ones(state_dim, dtype=np.float32)

    normalized_data = dict(data)
    normalized_data["observations"] = normalize_states(data["observations"], state_mean, state_std)
    normalized_data["next_observations"] = normalize_states(data["next_observations"], state_mean, state_std)

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=max(buffer_size, normalized_data["observations"].shape[0]),
        device=device,
    )
    replay_buffer.load_dataset(normalized_data)

    eval_env = _make_eval_env(env_name, state_dim, action_dim)
    max_action = float(getattr(eval_env.action_space, "high", np.ones(action_dim, dtype=np.float32))[0])
    actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    trainer = BCTrainer(actor=actor, actor_optimizer=actor_optimizer)
    eval_policy = NormalizedActorAdapter(actor, state_mean=state_mean, state_std=state_std).to(device)

    config = {
        "method": "corl-bc",
        "env_name": env_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": batch_size,
        "max_timesteps": max_timesteps,
        "eval_freq": eval_freq,
        "seed": seed,
        "device": device,
        "mix_aug_ratio": mix_aug_ratio,
        "frac": frac,
        "discount": discount,
        "normalize": normalize,
    }
    logger = ExperimentLogger(
        save_dir=save_dir,
        use_wandb=use_wandb,
        project=wandb_project,
        group=wandb_group,
        name=wandb_name,
        config=config,
    )

    train_metrics: Dict[str, float] = {}
    eval_metrics: Dict[str, float] = {}

    for step in range(int(max_timesteps)):
        train_metrics = trainer.train(replay_buffer.sample(batch_size))
        if (step + 1) % max(1, log_every) == 0:
            logger.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=step + 1)
            print(_format_train(step + 1, int(max_timesteps), train_metrics), flush=True)
        if (step + 1) % max(1, eval_freq) == 0 or step == int(max_timesteps) - 1:
            eval_metrics = evaluate_policy(eval_env, eval_policy, device=device, n_episodes=eval_episodes)
            logger.log_metrics({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step + 1)
            print(_format_eval(step + 1, int(max_timesteps), eval_metrics), flush=True)

    checkpoint = trainer.state_dict()
    checkpoint["state_mean"] = state_mean
    checkpoint["state_std"] = state_std
    torch.save(checkpoint, save_dir / "checkpoint.pt")
    logger.write_eval(eval_metrics)
    logger.finish()
    return {"train": train_metrics, "eval": eval_metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CORL-style BC training for VGKS trajectories")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument("--raw-dataset-path", dest="raw_dataset_path", type=str, default=None)
    parser.add_argument("--aug-dataset-path", dest="aug_dataset_path", type=str, default=None)
    parser.add_argument("--mix-aug-ratio", dest="mix_aug_ratio", type=float, default=0.0)
    parser.add_argument("--env-name", dest="env_name", type=str, default=None)
    parser.add_argument("--task", dest="task", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--state-dim", dest="state_dim", type=int, default=None)
    parser.add_argument("--action-dim", dest="action_dim", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--max-timesteps", dest="max_timesteps", type=int, default=int(1e6))
    parser.add_argument("--eval-freq", dest="eval_freq", type=int, default=5000)
    parser.add_argument("--log-every", dest="log_every", type=int, default=1000)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--device", dest="device", type=str, default="cuda")
    parser.add_argument("--save-dir", dest="save_dir", type=str, required=True)
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb-project", dest="wandb_project", type=str, default="vgks")
    parser.add_argument("--wandb-group", dest="wandb_group", type=str, default="corl-bc")
    parser.add_argument("--wandb-name", dest="wandb_name", type=str, default="corl-bc-run")
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=10)
    parser.add_argument("--frac", dest="frac", type=float, default=1.0)
    parser.add_argument("--discount", dest="discount", type=float, default=0.99)
    parser.add_argument("--buffer-size", dest="buffer_size", type=int, default=2_000_000)
    parser.add_argument("--max-episode-steps", dest="max_episode_steps", type=int, default=1000)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    resolved_env_name = resolve_env_name(args.env_name, args.task, args.dataset_name)
    state_dim = args.state_dim
    action_dim = args.action_dim
    if resolved_env_name is not None and (state_dim is None or action_dim is None):
        dims = infer_env_dims(make_env(resolved_env_name))
        state_dim = dims["state_dim"]
        action_dim = dims["action_dim"]
    if state_dim is None or action_dim is None:
        raise ValueError("state_dim and action_dim are required when env_name is not provided")

    metrics = run_corl_bc_training(
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        raw_dataset_path=Path(args.raw_dataset_path) if args.raw_dataset_path else None,
        aug_dataset_path=Path(args.aug_dataset_path) if args.aug_dataset_path else None,
        mix_aug_ratio=args.mix_aug_ratio,
        env_name=resolved_env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=args.batch_size,
        max_timesteps=args.max_timesteps,
        eval_freq=args.eval_freq,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
        save_dir=Path(args.save_dir),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        eval_episodes=args.eval_episodes,
        frac=args.frac,
        discount=args.discount,
        buffer_size=args.buffer_size,
        normalize=args.normalize,
        max_episode_steps=args.max_episode_steps,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
