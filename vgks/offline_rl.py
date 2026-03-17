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


def make_offline_loader(
    dataset_path: Optional[Path], env_name: Optional[str], batch_size: int, num_workers: int
):
    data = ensure_rewards_and_terminals(load_training_dataset(dataset_path, env_name))
    dataset = OfflineReplayDataset(data)
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data, loader


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
