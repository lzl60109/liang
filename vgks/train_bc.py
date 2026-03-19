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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


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


def _format_bc_train_progress(epoch: int, total_epochs: int, metrics: Dict[str, float]) -> str:
    return (
        f"[Train][BC] epoch={epoch}/{total_epochs} "
        f"bc_loss={metrics['bc_loss']:.4f} step_count={metrics['step_count']}"
    )


def _format_bc_eval_progress(metrics: Dict[str, float]) -> str:
    return (
        f"[Eval][BC] return={metrics['raw_return']:.4f} "
        f"normalized_score={metrics['normalized_score']:.4f} episodes={metrics['episodes']:.4f}"
    )


def train_bc_epoch(
    policy: BCPolicy,
    loader: DataLoader,
    optimizer,
    device: str,
    *,
    epoch: int = 1,
    total_epochs: int = 1,
    log_every: int = 0,
) -> Dict[str, float]:
    policy.train()
    total_loss = 0.0
    batch_count = 0
    iterator = loader
    progress = None
    if tqdm is not None:
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"[Train][BC] epoch={epoch}/{total_epochs}",
            file=sys.stdout,
            leave=False,
        )
        iterator = progress

    for batch in iterator:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        predicted = policy(observations)
        loss = torch.mean((predicted - actions) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        batch_count += 1
        if progress is not None:
            progress.set_postfix({"bc_loss": f"{total_loss / max(1, batch_count):.4f}"})
        elif log_every and batch_count % max(1, log_every) == 0:
            print(
                f"[Train][BC] epoch={epoch}/{total_epochs} batch={batch_count}/{len(loader)} "
                f"bc_loss={total_loss / max(1, batch_count):.4f}",
                flush=True,
            )
    metrics = {"bc_loss": total_loss / max(1, batch_count), "step_count": batch_count}
    print(_format_bc_train_progress(epoch, total_epochs, metrics), flush=True)
    return metrics


def _make_eval_env(env_name: Optional[str], state_dim: int, action_dim: int):
    if env_name is None:
        return ToyEvalEnv(state_dim, action_dim)
    return make_env(env_name)


def build_bc_training_data(
    *,
    dataset_path: Optional[Path] = None,
    raw_dataset_path: Optional[Path] = None,
    aug_dataset_path: Optional[Path] = None,
    env_name: Optional[str] = None,
    mix_aug_ratio: float = 0.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    if dataset_path is not None:
        return load_offline_dataset(dataset_path)

    if raw_dataset_path is None:
        return load_d4rl_dataset(env_name)

    raw_data = load_offline_dataset(raw_dataset_path)
    if aug_dataset_path is None or mix_aug_ratio <= 0.0:
        return raw_data

    aug_data = load_offline_dataset(aug_dataset_path)
    raw_size = int(raw_data["observations"].shape[0])
    aug_size = int(aug_data["observations"].shape[0])
    take = min(aug_size, int(round(raw_size * mix_aug_ratio)))
    if take <= 0:
        return raw_data

    rng = np.random.default_rng(seed)
    indices = rng.choice(aug_size, size=take, replace=False)
    mixed = {}
    for key in ("observations", "actions", "next_observations"):
        left = np.asarray(raw_data[key], dtype=np.float32)
        right = np.asarray(aug_data[key], dtype=np.float32)[indices]
        mixed[key] = np.concatenate([left, right], axis=0)
    return mixed


def run_bc_training(
    *,
    dataset_path: Optional[Path],
    raw_dataset_path: Optional[Path] = None,
    aug_dataset_path: Optional[Path] = None,
    mix_aug_ratio: float = 0.0,
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
    log_every: int = 0,
) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = build_bc_training_data(
        dataset_path=dataset_path,
        raw_dataset_path=raw_dataset_path,
        aug_dataset_path=aug_dataset_path,
        env_name=env_name,
        mix_aug_ratio=mix_aug_ratio,
        seed=seed,
    )

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
        "mix_aug_ratio": mix_aug_ratio,
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
        epoch_metrics = train_bc_epoch(
            policy,
            loader,
            optimizer,
            device,
            epoch=epoch + 1,
            total_epochs=epochs,
            log_every=log_every,
        )
        logger.log_metrics(epoch_metrics, step=epoch + 1)

    eval_env = _make_eval_env(env_name, state_dim, action_dim)
    eval_metrics = evaluate_policy(eval_env, policy, device=device, n_episodes=eval_episodes)
    print(_format_bc_eval_progress(eval_metrics), flush=True)
    logger.write_eval(eval_metrics)
    torch.save(policy.state_dict(), save_dir / "checkpoint.pt")
    logger.finish()

    return {"train": epoch_metrics, "eval": eval_metrics}


def main() -> None:
    parser = build_parser()
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument("--raw-dataset-path", dest="raw_dataset_path", type=str, default=None)
    parser.add_argument("--aug-dataset-path", dest="aug_dataset_path", type=str, default=None)
    parser.add_argument("--mix-aug-ratio", dest="mix_aug_ratio", type=float, default=0.0)
    parser.add_argument("--env-name", dest="env_name", type=str, default=None)
    parser.add_argument("--task", dest="task", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--state-dim", dest="state_dim", type=int, default=None)
    parser.add_argument("--action-dim", dest="action_dim", type=int, default=None)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--epochs", dest="epochs", type=int, default=50)
    parser.add_argument("--log-every", dest="log_every", type=int, default=0)
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
        raw_dataset_path=Path(args.raw_dataset_path) if args.raw_dataset_path else None,
        aug_dataset_path=Path(args.aug_dataset_path) if args.aug_dataset_path else None,
        mix_aug_ratio=args.mix_aug_ratio,
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
        log_every=args.log_every,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
