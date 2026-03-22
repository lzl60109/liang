from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vgks.data import OfflineReplayDataset, build_dataloader
from vgks.envs import infer_env_dims, make_env, resolve_env_name
from vgks.eval import evaluate_policy
from vgks.train_vgks import infer_dims_from_dataset_source, load_config_file, merge_config_with_args
from vgks.offline_rl import (
    DeterministicActor,
    TwinQ,
    ValueNet,
    build_td3bc_training_data,
    compute_state_stats,
    format_eval_progress,
    format_train_progress,
    infinite_batches,
    make_eval_env,
    resolve_total_steps,
    save_training_outputs,
    iql_train_step,
)
from vgks.experiment_logging import ExperimentLogger


def run_iql_training(
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
    epochs: Optional[int] = None,
    max_timesteps: Optional[int] = None,
    eval_freq: int = 5000,
    log_every: int = 1000,
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

    data = build_td3bc_training_data(
        dataset_path=dataset_path,
        raw_dataset_path=raw_dataset_path,
        aug_dataset_path=aug_dataset_path,
        env_name=env_name,
        mix_aug_ratio=mix_aug_ratio,
        seed=seed,
    )
    dataset = OfflineReplayDataset(data)
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    state_stats = compute_state_stats(data)
    logger = ExperimentLogger(
        save_dir=save_dir,
        use_wandb=use_wandb,
        project=wandb_project,
        group=wandb_group,
        name=wandb_name,
        config={
            "method": "iql",
            "env_name": env_name,
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "raw_dataset_path": str(raw_dataset_path) if raw_dataset_path is not None else None,
            "aug_dataset_path": str(aug_dataset_path) if aug_dataset_path is not None else None,
            "mix_aug_ratio": mix_aug_ratio,
            "epochs": epochs,
            "max_timesteps": max_timesteps,
            "eval_freq": eval_freq,
            "log_every": log_every,
            "seed": seed,
            "device": device,
        },
    )

    actor = DeterministicActor(state_dim, action_dim, hidden_dim).to(device)
    critic = TwinQ(state_dim, action_dim, hidden_dim).to(device)
    value_net = ValueNet(state_dim, hidden_dim).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    eval_env = make_eval_env(env_name, state_dim, action_dim)
    total_steps = resolve_total_steps(epochs, max_timesteps, loader)
    train_metrics = {}
    eval_metrics = {}
    batch_iterator = infinite_batches(loader)
    for step in range(total_steps):
        train_metrics = iql_train_step(
            actor, critic, value_net, actor_optimizer, critic_optimizer, value_optimizer, next(batch_iterator), device
        )
        if (step + 1) % max(1, log_every) == 0:
            logger.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=step + 1)
            print(format_train_progress("iql", step=step + 1, total_steps=total_steps, metrics=train_metrics), flush=True)
        if (step + 1) % max(1, eval_freq) == 0 or step == total_steps - 1:
            eval_metrics = evaluate_policy(eval_env, actor, device=device, n_episodes=eval_episodes)
            logger.log_metrics({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step + 1)
            print(format_eval_progress("iql", step=step + 1, total_steps=total_steps, metrics=eval_metrics), flush=True)

    save_training_outputs(
        logger,
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "value_net": value_net.state_dict(),
            "data_keys": list(data.keys()),
            "state_mean": state_stats["state_mean"],
            "state_std": state_stats["state_std"],
        },
        save_dir,
        eval_metrics,
    )
    return {"train": train_metrics, "eval": eval_metrics}


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description="Train IQL on offline data")
    parser.add_argument("--config", dest="config", type=str, default=None)
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
    parser.add_argument("--epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--max-timesteps", dest="max_timesteps", type=int, default=None)
    parser.add_argument("--eval-freq", dest="eval_freq", type=int, default=None)
    parser.add_argument("--log-every", dest="log_every", type=int, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--save-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb-project", dest="wandb_project", type=str, default="vgks")
    parser.add_argument("--wandb-group", dest="wandb_group", type=str, default="iql")
    parser.add_argument("--wandb-name", dest="wandb_name", type=str, default="iql-run")
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=10)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    parser.set_defaults(
        dataset_path=None,
        raw_dataset_path=None,
        aug_dataset_path=None,
        mix_aug_ratio=None,
        env_name=None,
        task=None,
        dataset_name=None,
        state_dim=None,
        action_dim=None,
        hidden_dim=None,
        batch_size=None,
        epochs=None,
        max_timesteps=None,
        eval_freq=None,
        log_every=None,
        seed=None,
        device=None,
        save_dir=None,
        use_wandb=None,
        wandb_project=None,
        wandb_group=None,
        wandb_name=None,
        eval_episodes=None,
        num_workers=None,
    )
    args = parser.parse_args()

    config = load_config_file(Path(args.config)) if args.config else {}
    merged = merge_config_with_args(config, vars(args))
    resolved_env_name = resolve_env_name(merged.get("env_name"), merged.get("task"), merged.get("dataset_name"))
    dataset_path = Path(merged["dataset_path"]) if merged.get("dataset_path") else None
    state_dim = merged.get("state_dim")
    action_dim = merged.get("action_dim")
    if dataset_path is not None and (state_dim is None or action_dim is None):
        dims = infer_dims_from_dataset_source(dataset_path)
        state_dim = dims["state_dim"]
        action_dim = dims["action_dim"]
    if resolved_env_name is not None and (state_dim is None or action_dim is None):
        dims = infer_env_dims(make_env(resolved_env_name))
        state_dim = dims["state_dim"]
        action_dim = dims["action_dim"]
    if state_dim is None or action_dim is None:
        raise ValueError("state_dim and action_dim are required when env_name is not provided")
    if merged.get("save_dir") is None:
        raise ValueError("save_dir is required")

    metrics = run_iql_training(
        dataset_path=dataset_path,
        raw_dataset_path=Path(merged["raw_dataset_path"]) if merged.get("raw_dataset_path") else None,
        aug_dataset_path=Path(merged["aug_dataset_path"]) if merged.get("aug_dataset_path") else None,
        mix_aug_ratio=merged.get("mix_aug_ratio", 0.0),
        env_name=resolved_env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=merged["hidden_dim"],
        batch_size=merged["batch_size"],
        epochs=merged.get("epochs"),
        max_timesteps=merged.get("max_timesteps"),
        eval_freq=merged.get("eval_freq", 5000),
        log_every=merged.get("log_every", 1000),
        seed=merged["seed"],
        device=merged["device"],
        save_dir=Path(merged["save_dir"]),
        use_wandb=bool(merged["use_wandb"]),
        wandb_project=merged["wandb_project"],
        wandb_group=merged["wandb_group"],
        wandb_name=merged["wandb_name"],
        eval_episodes=merged["eval_episodes"],
        num_workers=merged["num_workers"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
