from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vgks.cli import build_parser
from vgks.data import OfflineReplayDataset, build_dataloader, load_d4rl_dataset, load_offline_dataset
from vgks.envs import infer_env_dims, make_env, resolve_env_name
from vgks.eval import evaluate_policy
from vgks.experiment_logging import ExperimentLogger


class BCPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.float())


class ToyEvalEnv:
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.observation_space = type("Space", (), {"shape": (state_dim,)})()
        self.action_space = type("Space", (), {"shape": (action_dim,)})()
        self._step = 0

    def reset(self):
        self._step = 0
        return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def step(self, action):
        self._step += 1
        reward = 1.0
        done = self._step >= 3
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return obs, reward, done, {}

    def get_normalized_score(self, raw_return):
        return raw_return / 3.0


def train_bc_epoch(policy: BCPolicy, loader: DataLoader, optimizer, device: str) -> Dict[str, float]:
    policy.train()
    total_loss = 0.0
    batch_count = 0
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        predicted = policy(observations)
        loss = torch.mean((predicted - actions) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        batch_count += 1
    return {"bc_loss": total_loss / max(1, batch_count), "step_count": batch_count}


def _make_eval_env(env_name: Optional[str], state_dim: int, action_dim: int):
    if env_name is None:
        return ToyEvalEnv(state_dim, action_dim)
    return make_env(env_name)


def run_bc_training(
    *,
    dataset_path: Optional[Path],
    env_name: Optional[str],
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    seed: int,
    device: str,
    save_dir: Path,
    use_wandb: bool,
    wandb_project: str,
    wandb_group: str,
    wandb_name: str,
    eval_episodes: int = 10,
    num_workers: int = 0,
) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_path is not None:
        data = load_offline_dataset(dataset_path)
    else:
        data = load_d4rl_dataset(env_name)

    dataset = OfflineReplayDataset(data)
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    config = {
        "method": "bc",
        "env_name": env_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "device": device,
    }
    logger = ExperimentLogger(
        save_dir=save_dir,
        use_wandb=use_wandb,
        project=wandb_project,
        group=wandb_group,
        name=wandb_name,
        config=config,
    )

    policy = BCPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    epoch_metrics = {}
    for epoch in range(epochs):
        epoch_metrics = train_bc_epoch(policy, loader, optimizer, device)
        logger.log_metrics(epoch_metrics, step=epoch + 1)

    eval_env = _make_eval_env(env_name, state_dim, action_dim)
    eval_metrics = evaluate_policy(eval_env, policy, device=device, n_episodes=eval_episodes)
    logger.write_eval(eval_metrics)
    torch.save(policy.state_dict(), save_dir / "checkpoint.pt")
    logger.finish()

    return {"train": epoch_metrics, "eval": eval_metrics}


def main() -> None:
    parser = build_parser()
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument("--env-name", dest="env_name", type=str, default=None)
    parser.add_argument("--task", dest="task", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--state-dim", dest="state_dim", type=int, default=None)
    parser.add_argument("--action-dim", dest="action_dim", type=int, default=None)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--epochs", dest="epochs", type=int, default=50)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--save-dir", dest="save_dir", type=str, required=True)
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb-project", dest="wandb_project", type=str, default="vgks")
    parser.add_argument("--wandb-group", dest="wandb_group", type=str, default="bc")
    parser.add_argument("--wandb-name", dest="wandb_name", type=str, default="bc-run")
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=10)
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

    metrics = run_bc_training(
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        env_name=resolved_env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        save_dir=Path(args.save_dir),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        eval_episodes=args.eval_episodes,
        num_workers=args.num_workers,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
