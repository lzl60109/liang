from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vgks.data import OfflineReplayDataset, build_dataloader, load_d4rl_dataset, load_offline_dataset
from vgks.envs import infer_env_dims, make_env, resolve_env_name
from vgks.eval import evaluate_policy
from vgks.export import export_augmented_dataset
from vgks.experiment_logging import ExperimentLogger
from vgks.models import ConservativeCritic, InverseDynamicsModel, KoopmanDynamicsModel, SigmaModel
from vgks.train_bc import BCPolicy, ToyEvalEnv, train_bc_epoch
from vgks.trainer import ValueGuidedKoopmanTrainer


def _make_eval_env(env_name: Optional[str], state_dim: int, action_dim: int):
    if env_name is None:
        return ToyEvalEnv(state_dim, action_dim)
    return make_env(env_name)


def _train_koopman_components(
    *,
    dynamics: KoopmanDynamicsModel,
    inverse_model: InverseDynamicsModel,
    loader,
    device: str,
    epochs: int,
) -> None:
    params = list(dynamics.parameters()) + list(inverse_model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    for _ in range(epochs):
        for batch in loader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            next_observations = batch["next_observations"].to(device)

            z_t = dynamics.encode(observations)
            z_t1 = dynamics.encode(next_observations)
            predicted_next_latent = dynamics.predict_next_latent(z_t)
            predicted_next_state = dynamics.decode(predicted_next_latent)
            reconstructed_actions = inverse_model(z_t, z_t1)

            loss = F.mse_loss(predicted_next_state, next_observations) + F.mse_loss(
                reconstructed_actions, actions
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _concatenate_augmented_batches(batches):
    keys = batches[0].keys()
    merged = {}
    for key in keys:
        values = [batch[key] for batch in batches]
        first = values[0]
        if isinstance(first, torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = sum(int(value) for value in values)
    return merged


def run_kats_training(
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
        "method": "kats",
        "env_name": env_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
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

    dynamics = KoopmanDynamicsModel(state_dim, action_dim, latent_dim, hidden_dim).to(device)
    inverse_model = InverseDynamicsModel(latent_dim, action_dim, hidden_dim).to(device)
    _train_koopman_components(
        dynamics=dynamics,
        inverse_model=inverse_model,
        loader=loader,
        device=device,
        epochs=max(1, epochs),
    )

    sigma_model = SigmaModel(latent_dim).to(device)
    dummy_critic = ConservativeCritic(state_dim, action_dim, hidden_dim).to(device)
    trainer = ValueGuidedKoopmanTrainer(
        dynamics=dynamics,
        sigma_model=sigma_model,
        critic=dummy_critic,
        inverse_model=inverse_model,
        action_dim=action_dim,
        lambda_q=0.0,
        lambda_state_anchor=1.0,
        lambda_latent_anchor=0.1,
        sigma_lr=1e-3,
        device=torch.device(device),
    )

    sigma_metrics = {}
    for epoch in range(epochs):
        sigma_metrics = trainer.train_sigma_epoch(loader)
        logger.log_metrics({f"sigma/{k}": v for k, v in sigma_metrics.items() if k != "step_count"}, step=epoch + 1)

    augmented_batches = [trainer.augment_batch(batch) for batch in loader]
    augmented = _concatenate_augmented_batches(augmented_batches)
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

    policy = BCPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    train_metrics = {}
    for epoch in range(epochs):
        train_metrics = train_bc_epoch(policy, combined_loader, optimizer, device)
        logger.log_metrics({f"policy/{k}": v for k, v in train_metrics.items()}, step=epoch + 1)

    eval_env = _make_eval_env(env_name, state_dim, action_dim)
    eval_metrics = evaluate_policy(eval_env, policy, device=device, n_episodes=eval_episodes)
    logger.write_eval(eval_metrics)
    torch.save(
        {
            "policy": policy.state_dict(),
            "dynamics": dynamics.state_dict(),
            "inverse_model": inverse_model.state_dict(),
            "sigma_model": sigma_model.state_dict(),
        },
        save_dir / "checkpoint.pt",
    )
    logger.finish()

    return {"train": train_metrics, "sigma": sigma_metrics, "eval": eval_metrics}


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description="KATS baseline")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument("--env-name", dest="env_name", type=str, default=None)
    parser.add_argument("--task", dest="task", type=str, default=None)
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None)
    parser.add_argument("--state-dim", dest="state_dim", type=int, default=None)
    parser.add_argument("--action-dim", dest="action_dim", type=int, default=None)
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=32)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--save-dir", dest="save_dir", type=str, required=True)
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb-project", dest="wandb_project", type=str, default="vgks")
    parser.add_argument("--wandb-group", dest="wandb_group", type=str, default="kats")
    parser.add_argument("--wandb-name", dest="wandb_name", type=str, default="kats-run")
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

    metrics = run_kats_training(
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        env_name=resolved_env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
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
