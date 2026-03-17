from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vgks.cli import build_parser
from vgks.data import OfflineReplayDataset, build_dataloader, load_d4rl_dataset, load_offline_dataset
from vgks.envs import infer_env_dims, make_env, resolve_env_name
from vgks.eval import evaluate_policy
from vgks.export import export_augmented_dataset
from vgks.integration import load_kats_checkpoint, load_tgcvg_critic_checkpoint
from vgks.experiment_logging import ExperimentLogger
from vgks.models import ConservativeCritic, KoopmanDynamicsModel, SigmaModel
from vgks.train_bc import BCPolicy, ToyEvalEnv, train_bc_epoch
from vgks.trainer import ValueGuidedKoopmanTrainer


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "vgks.yaml"


def load_config_file(config_path: Path) -> Dict[str, object]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("VGKS config must be a YAML mapping")
    base_ref = data.pop("base_config", None)
    if base_ref:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = Path(config_path).parent / base_path
        base_data = load_config_file(base_path)
        base_data.update(data)
        return base_data
    return data


def merge_config_with_args(config: Dict[str, object], arg_values: Dict[str, object]) -> Dict[str, object]:
    merged = dict(config)
    for key, value in arg_values.items():
        if value is not None:
            merged[key] = value
    return merged


def build_trainer_from_args(
    *,
    state_dim: int,
    action_dim: int,
    latent_dim: int,
    hidden_dim: int,
    lambda_q: float,
    lambda_state_anchor: float,
    lambda_latent_anchor: float,
    q_clip_min: float,
    q_clip_max: float,
    sigma_warmup_steps: int,
    sigma_lr: float,
    kats_checkpoint: Optional[str],
    critic_checkpoint: Optional[str],
    device: str = "cpu",
) -> ValueGuidedKoopmanTrainer:
    dynamics = KoopmanDynamicsModel(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
    sigma_model = SigmaModel(latent_dim=latent_dim)
    critic = ConservativeCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=max(256, hidden_dim),
    )

    if kats_checkpoint:
        load_kats_checkpoint(dynamics, kats_checkpoint)
    if critic_checkpoint:
        load_tgcvg_critic_checkpoint(critic, critic_checkpoint)

    return ValueGuidedKoopmanTrainer(
        dynamics=dynamics,
        sigma_model=sigma_model,
        critic=critic,
        action_dim=action_dim,
        lambda_q=lambda_q,
        lambda_state_anchor=lambda_state_anchor,
        lambda_latent_anchor=lambda_latent_anchor,
        q_clip_min=q_clip_min,
        q_clip_max=q_clip_max,
        sigma_warmup_steps=sigma_warmup_steps,
        sigma_lr=sigma_lr,
        device=torch.device(device),
    )


def run_training(
    *,
    trainer: ValueGuidedKoopmanTrainer,
    dataset_path: Optional[Path] = None,
    env_name: Optional[str] = None,
    batch_size: int = 256,
    epochs: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    save_dir: Optional[Path] = None,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    seed: int,
    device: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_group: str,
    wandb_name: str,
    eval_episodes: int = 10,
    eval_interval: int = 1,
    save_best: bool = True,
    run_name: Optional[str] = None,
) -> Dict[str, object]:
    if dataset_path is None and env_name is None:
        raise ValueError("run_training requires either dataset_path or env_name")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_path is not None:
        data = load_offline_dataset(dataset_path)
    else:
        data = load_d4rl_dataset(env_name)

    dataset = OfflineReplayDataset(data)
    logger = None
    if save_dir is not None:
        logger = ExperimentLogger(
            save_dir=save_dir,
            use_wandb=use_wandb,
            project=wandb_project,
            group=wandb_group,
            name=wandb_name,
            config={
                "method": "vgks",
                "env_name": env_name,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden_dim": hidden_dim,
                "batch_size": batch_size,
                "epochs": epochs,
                "seed": seed,
                "device": device,
                "eval_interval": eval_interval,
                "run_name": run_name,
            },
        )

    sigma_history: List[Dict[str, float]] = []
    eval_history: List[Dict[str, float]] = []
    best_eval: Optional[Dict[str, float]] = None
    best_normalized_score = float("-inf")
    policy = BCPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy_metrics = {}

    for epoch in range(epochs):
        loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        epoch_metrics = trainer.train_sigma_epoch(loader)
        sigma_history.append(epoch_metrics)
        if logger is not None:
            logger.log_metrics({f"sigma/{k}": v for k, v in epoch_metrics.items() if k != "step_count"}, step=epoch + 1)

        final_loader = build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        augmented_batches = [trainer.augment_batch(batch) for batch in final_loader]
        augmented = {
            key: torch.cat([batch[key] for batch in augmented_batches], dim=0)
            for key in augmented_batches[0].keys()
        }

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
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        policy_metrics = train_bc_epoch(policy, combined_loader, optimizer, device)
        if logger is not None:
            logger.log_metrics({f"policy/{k}": v for k, v in policy_metrics.items()}, step=epoch + 1)

        if (epoch + 1) % max(1, eval_interval) == 0 or epoch == epochs - 1:
            if env_name is None:
                eval_env = ToyEvalEnv(state_dim, action_dim)
            else:
                from vgks.envs import make_env

                eval_env = make_env(env_name)
            eval_metrics = evaluate_policy(eval_env, policy, device=device, n_episodes=eval_episodes)
            eval_entry = {"epoch": epoch + 1, **eval_metrics}
            eval_history.append(eval_entry)

            if eval_metrics["normalized_score"] > best_normalized_score:
                best_normalized_score = eval_metrics["normalized_score"]
                best_eval = eval_entry
                if save_dir is not None and save_best:
                    torch.save(
                        {
                            "sigma_model": trainer.sigma_model.state_dict(),
                            "policy": policy.state_dict(),
                            "epoch": epoch + 1,
                            "eval": eval_metrics,
                        },
                        save_dir / "best_checkpoint.pt",
                    )

            progress_line = (
                f"[Eval] epoch={epoch + 1} "
                f"sigma_loss={epoch_metrics['total_loss']:.6f} "
                f"policy_loss={policy_metrics['bc_loss']:.6f} "
                f"return={eval_metrics['raw_return']:.3f} "
                f"normalized_score={eval_metrics['normalized_score']:.3f} "
                f"best={best_normalized_score:.3f}"
            )
            print(progress_line)
            if logger is not None:
                logger.log_metrics(
                    {
                        "eval/raw_return": eval_metrics["raw_return"],
                        "eval/normalized_score": eval_metrics["normalized_score"],
                        "eval/best_normalized_score": best_normalized_score,
                    },
                    step=epoch + 1,
                )
                logger.log_text(progress_line)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(sigma_history, handle, indent=2)
        with open(save_dir / "eval_history.json", "w", encoding="utf-8") as handle:
            json.dump(eval_history, handle, indent=2)
        export_augmented_dataset(save_dir / "augmented_dataset.npz", augmented)
        torch.save(
            {
                "sigma_model": trainer.sigma_model.state_dict(),
                "policy": policy.state_dict(),
            },
            save_dir / "last_checkpoint.pt",
        )
        torch.save(
            {
                "sigma_model": trainer.sigma_model.state_dict(),
                "policy": policy.state_dict(),
            },
            save_dir / "checkpoint.pt",
        )
        if logger is not None:
            final_eval = eval_history[-1] if eval_history else {"raw_return": 0.0, "normalized_score": 0.0, "episodes": eval_episodes}
            logger.write_eval(final_eval)
            logger.finish()

    return {
        "sigma_history": sigma_history,
        "eval_history": eval_history,
        "best_normalized_score": best_normalized_score,
        "best_eval": best_eval,
        "last": {
            "sigma": sigma_history[-1],
            "policy": policy_metrics,
            "eval": eval_history[-1] if eval_history else None,
        },
    }


def main() -> None:
    parser = build_parser()
    parser.add_argument("--config", dest="config", type=str, default=None)
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument("--env-name", dest="env_name", type=str, default=None)
    parser.add_argument("--task", dest="task", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--state-dim", dest="state_dim", type=int, default=None)
    parser.add_argument("--action-dim", dest="action_dim", type=int, default=None)
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=32)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--epochs", dest="epochs", type=int, default=1)
    parser.add_argument("--sigma-lr", dest="sigma_lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb-project", dest="wandb_project", type=str, default="vgks")
    parser.add_argument("--wandb-group", dest="wandb_group", type=str, default="vgks")
    parser.add_argument("--wandb-name", dest="wandb_name", type=str, default="vgks-run")
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=10)
    parser.add_argument("--eval-interval", dest="eval_interval", type=int, default=1)
    parser.add_argument("--save-best", dest="save_best", action="store_true")
    parser.add_argument("--run-name", dest="run_name", type=str, default=None)
    parser.set_defaults(
        dataset_path=None,
        env_name=None,
        task=None,
        dataset_name=None,
        state_dim=None,
        action_dim=None,
        latent_dim=None,
        hidden_dim=None,
        batch_size=None,
        epochs=None,
        sigma_lr=None,
        save_dir=None,
        seed=None,
        use_wandb=None,
        wandb_project=None,
        wandb_group=None,
        wandb_name=None,
        eval_episodes=None,
        eval_interval=None,
        save_best=None,
        run_name=None,
        device=None,
        num_workers=None,
        lambda_q=None,
        lambda_state_anchor=None,
        lambda_latent_anchor=None,
        q_clip_min=None,
        q_clip_max=None,
        q_threshold=None,
        sigma_warmup_steps=None,
        kats_checkpoint=None,
        critic_checkpoint=None,
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    config = load_config_file(config_path) if config_path.exists() else {}
    merged = merge_config_with_args(config, vars(args))

    resolved_env_name = resolve_env_name(merged.get("env_name"), merged.get("task"), merged.get("dataset_name"))
    state_dim = merged.get("state_dim")
    action_dim = merged.get("action_dim")
    if resolved_env_name is not None and (state_dim is None or action_dim is None):
        dims = infer_env_dims(make_env(resolved_env_name))
        state_dim = dims["state_dim"]
        action_dim = dims["action_dim"]
    if state_dim is None or action_dim is None:
        raise ValueError("state_dim and action_dim are required when env_name is not provided")

    trainer = build_trainer_from_args(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=merged["latent_dim"],
        hidden_dim=merged["hidden_dim"],
        lambda_q=merged["lambda_q"],
        lambda_state_anchor=merged["lambda_state_anchor"],
        lambda_latent_anchor=merged["lambda_latent_anchor"],
        q_clip_min=merged["q_clip_min"],
        q_clip_max=merged["q_clip_max"],
        sigma_warmup_steps=merged["sigma_warmup_steps"],
        sigma_lr=merged["sigma_lr"],
        kats_checkpoint=merged.get("kats_checkpoint"),
        critic_checkpoint=merged.get("critic_checkpoint"),
        device=merged["device"],
    )
    metrics = run_training(
        trainer=trainer,
        dataset_path=Path(merged["dataset_path"]) if merged.get("dataset_path") else None,
        env_name=resolved_env_name,
        batch_size=merged["batch_size"],
        epochs=merged["epochs"],
        num_workers=merged["num_workers"],
        save_dir=Path(merged["save_dir"]) if merged.get("save_dir") else None,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=merged["hidden_dim"],
        seed=merged["seed"],
        device=merged["device"],
        use_wandb=bool(merged["use_wandb"]),
        wandb_project=merged["wandb_project"],
        wandb_group=merged["wandb_group"],
        wandb_name=merged["wandb_name"],
        eval_episodes=merged["eval_episodes"],
        eval_interval=merged["eval_interval"],
        save_best=bool(merged["save_best"]),
        run_name=merged.get("run_name"),
    )
    print(json.dumps(metrics["last"], indent=2))


if __name__ == "__main__":
    main()
