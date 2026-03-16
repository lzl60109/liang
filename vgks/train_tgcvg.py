from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from vgks.data import OfflineReplayDataset, build_dataloader, load_d4rl_dataset, load_offline_dataset
from vgks.envs import infer_env_dims, make_env, resolve_env_name
from vgks.eval import evaluate_policy
from vgks.export import export_augmented_dataset
from vgks.logging import ExperimentLogger
from vgks.models import ConservativeCritic
from vgks.train_bc import BCPolicy, ToyEvalEnv, train_bc_epoch


def _make_eval_env(env_name: Optional[str], state_dim: int, action_dim: int):
    if env_name is None:
        return ToyEvalEnv(state_dim, action_dim)
    return make_env(env_name)


def _train_critic(critic: ConservativeCritic, loader, device: str, epochs: int) -> None:
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for _ in range(epochs):
        for batch in loader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            next_observations = batch["next_observations"].to(device)
            target_reward = -torch.mean((next_observations - observations) ** 2, dim=1)
            q1 = critic.q1(observations, actions)
            q2 = critic.q2(observations, actions)
            loss = F.mse_loss(q1, target_reward) + F.mse_loss(q2, target_reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _generate_augmented_dataset(critic, policy, loader, device: str, num_candidates: int = 4):
    augmented_batches = []
    with torch.no_grad():
        for batch in loader:
            observations = batch["observations"].to(device)
            next_observations = batch["next_observations"].to(device)
            base_actions = policy(observations)
            candidates = []
            q_values = []
            for _ in range(num_candidates):
                candidate = base_actions + 0.05 * torch.randn_like(base_actions)
                q = critic.conservative_value(observations, candidate)
                candidates.append(candidate.unsqueeze(1))
                q_values.append(q.unsqueeze(1))
            candidate_actions = torch.cat(candidates, dim=1)
            candidate_q = torch.cat(q_values, dim=1)
            best_indices = candidate_q.argmax(dim=1)
            best_actions = candidate_actions[
                torch.arange(candidate_actions.shape[0], device=device), best_indices
            ]
            best_q = candidate_q[torch.arange(candidate_q.shape[0], device=device), best_indices]
            augmented_batches.append(
                {
                    "observations": observations.detach().cpu(),
                    "actions": best_actions.detach().cpu(),
                    "next_observations": next_observations.detach().cpu(),
                    "q_values": best_q.detach().cpu(),
                }
            )
    keys = augmented_batches[0].keys()
    return {key: torch.cat([batch[key] for batch in augmented_batches], dim=0) for key in keys}


def run_tgcvg_training(
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
        "method": "tgcvg",
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
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    pretrain_metrics = {}
    for epoch in range(max(1, epochs)):
        pretrain_metrics = train_bc_epoch(policy, loader, policy_optimizer, device)
        logger.log_metrics({f"pretrain/{k}": v for k, v in pretrain_metrics.items()}, step=epoch + 1)

    critic = ConservativeCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    _train_critic(critic, loader, device=device, epochs=max(1, epochs))

    augmented = _generate_augmented_dataset(critic, policy, loader, device=device)
    export_augmented_dataset(save_dir / "augmented_dataset.npz", augmented)

    combined = {
        "observations": torch.cat(
            [torch.tensor(data["observations"], dtype=torch.float32), augmented["observations"]], dim=0
        ),
        "actions": torch.cat(
            [torch.tensor(data["actions"], dtype=torch.float32), augmented["actions"]], dim=0
        ),
        "next_observations": torch.cat(
            [torch.tensor(data["next_observations"], dtype=torch.float32), augmented["next_observations"]],
            dim=0,
        ),
    }
    combined_dataset = OfflineReplayDataset({key: value.numpy() for key, value in combined.items()})
    combined_loader = build_dataloader(
        combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    train_metrics = {}
    for epoch in range(epochs):
        train_metrics = train_bc_epoch(policy, combined_loader, policy_optimizer, device)
        logger.log_metrics({f"policy/{k}": v for k, v in train_metrics.items()}, step=epoch + 1)

    eval_env = _make_eval_env(env_name, state_dim, action_dim)
    eval_metrics = evaluate_policy(eval_env, policy, device=device, n_episodes=eval_episodes)
    logger.write_eval(eval_metrics)
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()}, save_dir / "checkpoint.pt")
    logger.finish()

    return {"train": train_metrics, "critic": pretrain_metrics, "eval": eval_metrics}


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description="TGCVG baseline")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument("--env-name", dest="env_name", type=str, default=None)
    parser.add_argument("--task", dest="task", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--state-dim", dest="state_dim", type=int, default=None)
    parser.add_argument("--action-dim", dest="action_dim", type=int, default=None)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--save-dir", dest="save_dir", type=str, required=True)
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb-project", dest="wandb_project", type=str, default="vgks")
    parser.add_argument("--wandb-group", dest="wandb_group", type=str, default="tgcvg")
    parser.add_argument("--wandb-name", dest="wandb_name", type=str, default="tgcvg-run")
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=10)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=0)
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

    metrics = run_tgcvg_training(
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
