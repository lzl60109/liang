from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vgks.data import (
    OfflineReplayDataset,
    build_dataloader,
    load_d4rl_dataset,
    load_offline_dataset,
    replay_to_trajectory_list,
    save_trajectory_paths,
)
from vgks.envs import infer_env_dims, make_env, resolve_env_name
from vgks.experiment_logging import ExperimentLogger
from vgks.train_vgks import (
    DEFAULT_CONFIG_PATH,
    build_trainer_from_args,
    infer_dims_from_dataset_source,
    load_config_file,
    merge_config_with_args,
)


def _to_numpy_dict(batch_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    arrays = {}
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            arrays[key] = value.detach().cpu().numpy()
        else:
            arrays[key] = np.asarray(value, dtype=np.float32)
    return arrays


def _concat_batches(batches):
    keys = batches[0].keys()
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


def generate_augmented_dataset(
    *,
    trainer,
    dataset_path: Optional[Path],
    env_name: Optional[str],
    batch_size: int,
    epochs: int,
    num_workers: int = 0,
) -> Dict[str, torch.Tensor]:
    if dataset_path is not None:
        data = load_offline_dataset(dataset_path)
    else:
        data = load_d4rl_dataset(env_name)

    dataset = OfflineReplayDataset(data)
    train_loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for _ in range(max(1, epochs)):
        trainer.train_sigma_epoch(train_loader)

    augmented_batches = []
    offset = 0
    for batch in eval_loader:
        augmented = trainer.augment_batch(batch)
        batch_size_now = augmented["observations"].shape[0]
        if "rewards" in data and data["rewards"] is not None:
            augmented["rewards"] = torch.tensor(
                data["rewards"][offset : offset + batch_size_now], dtype=torch.float32
            )
        if "terminals" in data and data["terminals"] is not None:
            augmented["terminals"] = torch.tensor(
                data["terminals"][offset : offset + batch_size_now], dtype=torch.float32
            )
        augmented_batches.append(augmented)
        offset += batch_size_now

    return _concat_batches(augmented_batches)


def save_generated_dataset(output_dir: Path, dataset_name: str, augmented: Dict[str, torch.Tensor]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / dataset_name
    arrays = _to_numpy_dict(augmented)
    np.savez(prefix.with_suffix(".npz"), **arrays)
    save_trajectory_paths(output_dir, dataset_name, replay_to_trajectory_list(arrays))
    return prefix


def run_vgks_generation(
    *,
    dataset_path: Optional[Path],
    env_name: Optional[str],
    state_dim: int,
    action_dim: int,
    latent_dim: int,
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
    num_workers: int = 0,
    lambda_q: float = 0.1,
    lambda_state_anchor: float = 1.0,
    lambda_latent_anchor: float = 0.1,
    q_clip_min: float = -20.0,
    q_clip_max: float = 20.0,
    sigma_warmup_steps: int = 0,
    sigma_lr: float = 1e-3,
    kats_checkpoint: Optional[str] = None,
    critic_checkpoint: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = build_trainer_from_args(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lambda_q=lambda_q,
        lambda_state_anchor=lambda_state_anchor,
        lambda_latent_anchor=lambda_latent_anchor,
        q_clip_min=q_clip_min,
        q_clip_max=q_clip_max,
        sigma_warmup_steps=sigma_warmup_steps,
        sigma_lr=sigma_lr,
        kats_checkpoint=kats_checkpoint,
        critic_checkpoint=critic_checkpoint,
        device=device,
    )

    logger = ExperimentLogger(
        save_dir=save_dir,
        use_wandb=use_wandb,
        project=wandb_project,
        group=wandb_group,
        name=wandb_name,
        config={
            "stage": "generate_vgks",
            "env_name": env_name,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "device": device,
            "run_name": run_name,
        },
    )

    augmented = generate_augmented_dataset(
        trainer=trainer,
        dataset_path=dataset_path,
        env_name=env_name,
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
    )

    dataset_name = run_name or (env_name if env_name is not None else "augmented_dataset")
    prefix = save_generated_dataset(save_dir, dataset_name, augmented)
    metrics = {
        "num_samples": int(augmented["observations"].shape[0]),
        "output_prefix": str(prefix),
    }
    logger.write_eval(metrics)
    logger.finish()
    return metrics


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description="Generate VGKS augmented trajectories")
    parser.add_argument("--config", dest="config", type=str, default=None)
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument("--env-name", dest="env_name", type=str, default=None)
    parser.add_argument("--task", dest="task", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--state-dim", dest="state_dim", type=int, default=None)
    parser.add_argument("--action-dim", dest="action_dim", type=int, default=None)
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=None)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--sigma-lr", dest="sigma_lr", type=float, default=None)
    parser.add_argument("--save-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--device", dest="device", type=str, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb-project", dest="wandb_project", type=str, default=None)
    parser.add_argument("--wandb-group", dest="wandb_group", type=str, default=None)
    parser.add_argument("--wandb-name", dest="wandb_name", type=str, default=None)
    parser.add_argument("--run-name", dest="run_name", type=str, default=None)
    parser.add_argument("--lambda-q", dest="lambda_q", type=float, default=None)
    parser.add_argument("--lambda-state-anchor", dest="lambda_state_anchor", type=float, default=None)
    parser.add_argument("--lambda-latent-anchor", dest="lambda_latent_anchor", type=float, default=None)
    parser.add_argument("--q-clip-min", dest="q_clip_min", type=float, default=None)
    parser.add_argument("--q-clip-max", dest="q_clip_max", type=float, default=None)
    parser.add_argument("--sigma-warmup-steps", dest="sigma_warmup_steps", type=int, default=None)
    parser.add_argument("--kats-checkpoint", dest="kats_checkpoint", type=str, default=None)
    parser.add_argument("--critic-checkpoint", dest="critic_checkpoint", type=str, default=None)
    parser.set_defaults(use_wandb=None)
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    config = load_config_file(config_path) if config_path.exists() else {}
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

    metrics = run_vgks_generation(
        dataset_path=dataset_path,
        env_name=resolved_env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=merged["latent_dim"],
        hidden_dim=merged["hidden_dim"],
        batch_size=merged["batch_size"],
        epochs=merged["epochs"],
        seed=merged["seed"],
        device=merged["device"],
        save_dir=Path(merged["save_dir"]),
        use_wandb=bool(merged["use_wandb"]),
        wandb_project=merged["wandb_project"],
        wandb_group=merged["wandb_group"],
        wandb_name=merged["wandb_name"],
        num_workers=merged["num_workers"],
        lambda_q=merged["lambda_q"],
        lambda_state_anchor=merged["lambda_state_anchor"],
        lambda_latent_anchor=merged["lambda_latent_anchor"],
        q_clip_min=merged["q_clip_min"],
        q_clip_max=merged["q_clip_max"],
        sigma_warmup_steps=merged["sigma_warmup_steps"],
        sigma_lr=merged["sigma_lr"],
        kats_checkpoint=merged.get("kats_checkpoint"),
        critic_checkpoint=merged.get("critic_checkpoint"),
        run_name=merged.get("run_name"),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
