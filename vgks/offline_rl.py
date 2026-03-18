from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vgks.data import OfflineReplayDataset, build_dataloader, load_d4rl_dataset, load_offline_dataset
from vgks.envs import make_env
from vgks.eval import evaluate_policy
from vgks.experiment_logging import ExperimentLogger
from vgks.models import ConservativeCritic
from vgks.train_bc import ToyEvalEnv

TD3BC_TRANSITION_KEYS = {"observations", "actions", "next_observations", "rewards", "terminals"}


def _format_metric_items(metrics: Dict[str, float]) -> str:
    parts = []
    preferred_order = [
        "actor_loss",
        "critic_loss",
        "value_loss",
        "cql_loss",
        "raw_return",
        "normalized_score",
        "episodes",
    ]
    seen = set()
    for key in preferred_order:
        if key in metrics:
            seen.add(key)
            display_key = "return" if key == "raw_return" else key
            value = metrics[key]
            if value is None:
                continue
            if isinstance(value, (int, float, np.floating, np.integer)):
                parts.append(f"{display_key}={float(value):.4f}")
            else:
                parts.append(f"{display_key}={value}")
    for key, value in metrics.items():
        if key in seen:
            continue
        if value is None:
            continue
        if isinstance(value, (int, float, np.floating, np.integer)):
            parts.append(f"{key}={float(value):.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def format_train_progress(method: str, *, step: int, total_steps: int, metrics: Dict[str, float]) -> str:
    return f"[Train][{method.upper()}] step={step}/{total_steps} {_format_metric_items(metrics)}".strip()


def format_eval_progress(method: str, *, step: int, total_steps: int, metrics: Dict[str, float]) -> str:
    return f"[Eval][{method.upper()}] step={step}/{total_steps} {_format_metric_items(metrics)}".strip()


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.float())


class NormalizedPolicy(nn.Module):
    def __init__(self, actor: nn.Module, state_mean: np.ndarray, state_std: np.ndarray) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("state_mean", torch.tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(state_std, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations.float() - self.state_mean) / self.state_std
        return self.actor(normalized)


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = copy.deepcopy(self.q1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        sa = torch.cat([observations, actions], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([observations, actions], dim=1))

    def q2_value(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q2(torch.cat([observations, actions], dim=1))

    def conservative(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(observations, actions)
        return torch.min(q1, q2)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.float())


def load_training_dataset(dataset_path: Optional[Path], env_name: Optional[str]):
    if dataset_path is not None:
        return load_offline_dataset(dataset_path)
    return load_d4rl_dataset(env_name)


def ensure_rewards_and_terminals(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normalized = dict(data)
    num_steps = normalized["observations"].shape[0]
    if normalized.get("rewards") is None:
        normalized["rewards"] = np.zeros(num_steps, dtype=np.float32)
    if normalized.get("terminals") is None:
        normalized["terminals"] = np.zeros(num_steps, dtype=np.float32)
    return normalized


def compute_state_stats(data: Dict[str, np.ndarray], eps: float = 1e-3) -> Dict[str, np.ndarray]:
    observations = np.asarray(data["observations"], dtype=np.float32)
    state_mean = observations.mean(axis=0).astype(np.float32)
    state_std = (observations.std(axis=0) + eps).astype(np.float32)
    return {"state_mean": state_mean, "state_std": state_std}


def make_offline_loader(
    dataset_path: Optional[Path], env_name: Optional[str], batch_size: int, num_workers: int
):
    data = ensure_rewards_and_terminals(load_training_dataset(dataset_path, env_name))
    dataset = OfflineReplayDataset(data)
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data, loader


def _concat_array_dicts(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    keys = sorted(set(left.keys()) | set(right.keys()))
    merged = {}
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None and right_value is None:
            merged[key] = None
        elif left_value is None:
            merged[key] = np.asarray(right_value, dtype=np.float32)
        elif right_value is None:
            merged[key] = np.asarray(left_value, dtype=np.float32)
        else:
            merged[key] = np.concatenate(
                [np.asarray(left_value, dtype=np.float32), np.asarray(right_value, dtype=np.float32)],
                axis=0,
            )
    return merged


def build_td3bc_training_data(
    *,
    dataset_path: Optional[Path] = None,
    raw_dataset_path: Optional[Path] = None,
    aug_dataset_path: Optional[Path] = None,
    env_name: Optional[str] = None,
    mix_aug_ratio: float = 0.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    if dataset_path is not None:
        data = ensure_rewards_and_terminals(load_training_dataset(dataset_path, env_name))
        return {key: value for key, value in data.items() if key in TD3BC_TRANSITION_KEYS}

    if raw_dataset_path is None:
        data = ensure_rewards_and_terminals(load_training_dataset(None, env_name))
        return {key: value for key, value in data.items() if key in TD3BC_TRANSITION_KEYS}

    raw_data = ensure_rewards_and_terminals(load_offline_dataset(raw_dataset_path))
    raw_data = {key: value for key, value in raw_data.items() if key in TD3BC_TRANSITION_KEYS}
    if aug_dataset_path is None or mix_aug_ratio <= 0.0:
        return raw_data

    aug_data = ensure_rewards_and_terminals(load_offline_dataset(aug_dataset_path))
    aug_data = {key: value for key, value in aug_data.items() if key in TD3BC_TRANSITION_KEYS}
    raw_size = int(raw_data["observations"].shape[0])
    aug_size = int(aug_data["observations"].shape[0])
    take = min(aug_size, int(round(raw_size * mix_aug_ratio)))
    if take <= 0:
        return raw_data

    rng = np.random.default_rng(seed)
    indices = rng.choice(aug_size, size=take, replace=False)
    aug_subset = {
        key: None if value is None else np.asarray(value, dtype=np.float32)[indices]
        for key, value in aug_data.items()
    }
    return _concat_array_dicts(raw_data, aug_subset)


def build_td3bc_training_sources(
    *,
    dataset_path: Optional[Path] = None,
    raw_dataset_path: Optional[Path] = None,
    aug_dataset_path: Optional[Path] = None,
    env_name: Optional[str] = None,
    mix_aug_ratio: float = 0.0,
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    critic_data = build_td3bc_training_data(
        dataset_path=dataset_path,
        raw_dataset_path=raw_dataset_path,
        aug_dataset_path=None,
        env_name=env_name,
        mix_aug_ratio=0.0,
        seed=seed,
    )
    actor_data = build_td3bc_training_data(
        dataset_path=dataset_path,
        raw_dataset_path=raw_dataset_path,
        aug_dataset_path=aug_dataset_path,
        env_name=env_name,
        mix_aug_ratio=mix_aug_ratio,
        seed=seed,
    )
    return {"critic_data": critic_data, "actor_data": actor_data}


def make_td3bc_loader(
    *,
    dataset_path: Optional[Path],
    raw_dataset_path: Optional[Path],
    aug_dataset_path: Optional[Path],
    env_name: Optional[str],
    mix_aug_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    data = build_td3bc_training_data(
        dataset_path=dataset_path,
        raw_dataset_path=raw_dataset_path,
        aug_dataset_path=aug_dataset_path,
        env_name=env_name,
        mix_aug_ratio=mix_aug_ratio,
        seed=seed,
    )
    dataset = OfflineReplayDataset(data)
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data, loader


def make_td3bc_dual_loaders(
    *,
    dataset_path: Optional[Path],
    raw_dataset_path: Optional[Path],
    aug_dataset_path: Optional[Path],
    env_name: Optional[str],
    mix_aug_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    sources = build_td3bc_training_sources(
        dataset_path=dataset_path,
        raw_dataset_path=raw_dataset_path,
        aug_dataset_path=aug_dataset_path,
        env_name=env_name,
        mix_aug_ratio=mix_aug_ratio,
        seed=seed,
    )
    critic_dataset = OfflineReplayDataset(sources["critic_data"])
    actor_dataset = OfflineReplayDataset(sources["actor_data"])
    critic_loader = build_dataloader(critic_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    actor_loader = build_dataloader(actor_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return sources, critic_loader, actor_loader


def infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch


def make_eval_env(env_name: Optional[str], state_dim: int, action_dim: int):
    if env_name is None:
        return ToyEvalEnv(state_dim, action_dim)
    return make_env(env_name)


def save_training_outputs(logger: ExperimentLogger, checkpoint: Dict[str, object], save_dir: Path, eval_metrics: Dict[str, float]):
    logger.write_eval(eval_metrics)
    torch.save(checkpoint, save_dir / "checkpoint.pt")
    logger.finish()


def resolve_total_steps(epochs: Optional[int], max_timesteps: Optional[int], loader) -> int:
    if max_timesteps is not None:
        return int(max_timesteps)
    if epochs is None:
        raise ValueError("Either epochs or max_timesteps must be provided")
    return int(max(1, epochs) * max(1, len(loader)))


class StableTD3BCTrainer:
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_action: float,
        device: str,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
    ) -> None:
        self.device = device
        self.max_action = float(max_action)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.total_it = 0
        self.state_mean = np.asarray(state_mean if state_mean is not None else np.zeros(state_dim), dtype=np.float32)
        self.state_std = np.asarray(state_std if state_std is not None else np.ones(state_dim), dtype=np.float32)
        self.state_mean_tensor = torch.tensor(self.state_mean, dtype=torch.float32, device=device)
        self.state_std_tensor = torch.tensor(self.state_std, dtype=torch.float32, device=device)

        self.actor = DeterministicActor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TwinQ(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _normalize(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations.float() - self.state_mean_tensor) / self.state_std_tensor

    def eval_policy(self) -> nn.Module:
        policy = NormalizedPolicy(self.actor, self.state_mean, self.state_std).to(self.device)
        policy.eval()
        return policy

    def _compute_q_mean(self, observations: torch.Tensor, actions: torch.Tensor) -> float:
        current_q1, current_q2 = self.critic(observations, actions)
        return float(torch.min(current_q1, current_q2).mean().detach().cpu().item())

    def train_critic_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        observations = self._normalize(batch["observations"].to(self.device))
        actions = batch["actions"].to(self.device).clamp(-self.max_action, self.max_action)
        next_observations = self._normalize(batch["next_observations"].to(self.device))
        rewards = batch["rewards"].to(self.device).unsqueeze(1)
        dones = batch["terminals"].to(self.device).unsqueeze(1)
        not_done = 1.0 - dones

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_observations) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + not_done * self.discount * target_q

        current_q1, current_q2 = self.critic(observations, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "q_mean": self._compute_q_mean(observations, actions),
        }

    def train_actor_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Optional[float]]:
        observations = self._normalize(batch["observations"].to(self.device))
        actions = batch["actions"].to(self.device).clamp(-self.max_action, self.max_action)
        pi = self.actor(observations)
        q_pi = self.critic.q1_value(observations, pi)
        lambda_coef = self.alpha / q_pi.abs().mean().detach().clamp(min=1e-6)
        bc_loss = F.mse_loss(pi, actions)
        actor_loss = -lambda_coef * q_pi.mean() + bc_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "bc_loss": float(bc_loss.detach().cpu().item()),
            "q_mean": self._compute_q_mean(observations, actions),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Optional[float]]:
        self.total_it += 1
        critic_metrics = self.train_critic_step(batch)
        actor_loss_value = None
        bc_loss_value = None
        q_mean = critic_metrics["q_mean"]
        if self.total_it % self.policy_freq == 0:
            actor_metrics = self.train_actor_step(batch)
            actor_loss_value = actor_metrics["actor_loss"]
            bc_loss_value = actor_metrics["bc_loss"]
            q_mean = actor_metrics["q_mean"]

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_metrics["critic_loss"],
            "bc_loss": bc_loss_value,
            "q_mean": q_mean,
        }


def run_td3bc_epoch(actor, critic, target_critic, actor_optimizer, critic_optimizer, loader, device: str, discount: float = 0.99):
    actor.train()
    critic.train()
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    batch_count = 0
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        next_observations = batch["next_observations"].to(device)
        rewards = batch["rewards"].to(device).unsqueeze(1)
        dones = batch["terminals"].to(device).unsqueeze(1)

        with torch.no_grad():
            next_actions = actor(next_observations)
            target_q = rewards + (1.0 - dones) * discount * target_critic.conservative(next_observations, next_actions)

        current_q1, current_q2 = critic(observations, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        predicted_actions = actor(observations)
        policy_q = critic.conservative(observations, predicted_actions)
        actor_loss = -policy_q.mean() + F.mse_loss(predicted_actions, actions)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.mul_(0.995).add_(0.005 * param.data)

        total_actor_loss += float(actor_loss.detach().cpu().item())
        total_critic_loss += float(critic_loss.detach().cpu().item())
        batch_count += 1

    return {
        "actor_loss": total_actor_loss / max(1, batch_count),
        "critic_loss": total_critic_loss / max(1, batch_count),
        "step_count": batch_count,
    }


def td3bc_train_step(actor, critic, target_critic, actor_optimizer, critic_optimizer, batch, device: str, discount: float = 0.99):
    actor.train()
    critic.train()
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    next_observations = batch["next_observations"].to(device)
    rewards = batch["rewards"].to(device).unsqueeze(1)
    dones = batch["terminals"].to(device).unsqueeze(1)

    with torch.no_grad():
        next_actions = actor(next_observations)
        target_q = rewards + (1.0 - dones) * discount * target_critic.conservative(next_observations, next_actions)

    current_q1, current_q2 = critic(observations, actions)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    predicted_actions = actor(observations)
    policy_q = critic.conservative(observations, predicted_actions)
    actor_loss = -policy_q.mean() + F.mse_loss(predicted_actions, actions)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.mul_(0.995).add_(0.005 * param.data)

    return {
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
    }


def run_iql_epoch(actor, critic, value_net, actor_optimizer, critic_optimizer, value_optimizer, loader, device: str, discount: float = 0.99, beta: float = 3.0):
    actor.train()
    critic.train()
    value_net.train()
    metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "value_loss": 0.0, "step_count": 0}
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        next_observations = batch["next_observations"].to(device)
        rewards = batch["rewards"].to(device)
        dones = batch["terminals"].to(device)

        with torch.no_grad():
            next_v = value_net(next_observations).squeeze(1)
            target_q = rewards + (1.0 - dones) * discount * next_v

        q1, q2 = critic(observations, actions)
        q_min = torch.min(q1, q2).squeeze(1)
        critic_loss = F.mse_loss(q1.squeeze(1), target_q) + F.mse_loss(q2.squeeze(1), target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        value = value_net(observations).squeeze(1)
        advantage = (q_min.detach() - value)
        weight = torch.where(advantage > 0, torch.tensor(0.7, device=device), torch.tensor(0.3, device=device))
        value_loss = torch.mean(weight * advantage.pow(2))
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        predicted_actions = actor(observations)
        actor_weight = torch.exp(beta * advantage.detach()).clamp(max=20.0).unsqueeze(1)
        actor_loss = torch.mean(actor_weight * (predicted_actions - actions).pow(2))
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        metrics["actor_loss"] += float(actor_loss.detach().cpu().item())
        metrics["critic_loss"] += float(critic_loss.detach().cpu().item())
        metrics["value_loss"] += float(value_loss.detach().cpu().item())
        metrics["step_count"] += 1

    steps = max(1, metrics["step_count"])
    metrics["actor_loss"] /= steps
    metrics["critic_loss"] /= steps
    metrics["value_loss"] /= steps
    return metrics


def iql_train_step(actor, critic, value_net, actor_optimizer, critic_optimizer, value_optimizer, batch, device: str, discount: float = 0.99, beta: float = 3.0):
    actor.train()
    critic.train()
    value_net.train()
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    next_observations = batch["next_observations"].to(device)
    rewards = batch["rewards"].to(device)
    dones = batch["terminals"].to(device)

    with torch.no_grad():
        next_v = value_net(next_observations).squeeze(1)
        target_q = rewards + (1.0 - dones) * discount * next_v

    q1, q2 = critic(observations, actions)
    q_min = torch.min(q1, q2).squeeze(1)
    critic_loss = F.mse_loss(q1.squeeze(1), target_q) + F.mse_loss(q2.squeeze(1), target_q)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    value = value_net(observations).squeeze(1)
    advantage = (q_min.detach() - value)
    weight = torch.where(advantage > 0, torch.tensor(0.7, device=device), torch.tensor(0.3, device=device))
    value_loss = torch.mean(weight * advantage.pow(2))
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    predicted_actions = actor(observations)
    actor_weight = torch.exp(beta * advantage.detach()).clamp(max=20.0).unsqueeze(1)
    actor_loss = torch.mean(actor_weight * (predicted_actions - actions).pow(2))
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return {
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
    }


def run_cql_epoch(actor, critic, actor_optimizer, critic_optimizer, loader, device: str, discount: float = 0.99, cql_alpha: float = 1.0):
    actor.train()
    critic.train()
    metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "cql_loss": 0.0, "step_count": 0}
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        next_observations = batch["next_observations"].to(device)
        rewards = batch["rewards"].to(device)
        dones = batch["terminals"].to(device)

        with torch.no_grad():
            next_actions = actor(next_observations)
            target_q = rewards + (1.0 - dones) * discount * critic.conservative_value(next_observations, next_actions)

        q1 = critic.q1(observations, actions)
        q2 = critic.q2(observations, actions)
        bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        random_actions = torch.empty_like(actions).uniform_(-1.0, 1.0)
        conservative_loss = (
            critic.q1(observations, random_actions).mean()
            + critic.q2(observations, random_actions).mean()
            - q1.mean()
            - q2.mean()
        )
        critic_loss = bellman_loss + cql_alpha * conservative_loss
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        predicted_actions = actor(observations)
        actor_loss = -critic.conservative_value(observations, predicted_actions).mean() + 0.5 * F.mse_loss(predicted_actions, actions)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        metrics["actor_loss"] += float(actor_loss.detach().cpu().item())
        metrics["critic_loss"] += float(bellman_loss.detach().cpu().item())
        metrics["cql_loss"] += float(conservative_loss.detach().cpu().item())
        metrics["step_count"] += 1

    steps = max(1, metrics["step_count"])
    metrics["actor_loss"] /= steps
    metrics["critic_loss"] /= steps
    metrics["cql_loss"] /= steps
    return metrics


def cql_train_step(actor, critic, actor_optimizer, critic_optimizer, batch, device: str, discount: float = 0.99, cql_alpha: float = 1.0):
    actor.train()
    critic.train()
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    next_observations = batch["next_observations"].to(device)
    rewards = batch["rewards"].to(device)
    dones = batch["terminals"].to(device)

    with torch.no_grad():
        next_actions = actor(next_observations)
        target_q = rewards + (1.0 - dones) * discount * critic.conservative_value(next_observations, next_actions)

    q1 = critic.q1(observations, actions)
    q2 = critic.q2(observations, actions)
    bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

    random_actions = torch.empty_like(actions).uniform_(-1.0, 1.0)
    conservative_loss = (
        critic.q1(observations, random_actions).mean()
        + critic.q2(observations, random_actions).mean()
        - q1.mean()
        - q2.mean()
    )
    critic_loss = bellman_loss + cql_alpha * conservative_loss
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    predicted_actions = actor(observations)
    actor_loss = -critic.conservative_value(observations, predicted_actions).mean() + 0.5 * F.mse_loss(predicted_actions, actions)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return {
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(bellman_loss.detach().cpu().item()),
        "cql_loss": float(conservative_loss.detach().cpu().item()),
    }
