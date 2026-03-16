from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from vgks.cli import build_parser
from vgks.data import OfflineReplayDataset, build_dataloader, load_d4rl_dataset, load_offline_dataset
from vgks.integration import load_kats_checkpoint, load_tgcvg_critic_checkpoint
from vgks.models import ConservativeCritic, KoopmanDynamicsModel, SigmaModel
from vgks.trainer import ValueGuidedKoopmanTrainer


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
    )


def run_training(
    *,
    trainer: ValueGuidedKoopmanTrainer,
    dataset_path: Optional[Path] = None,
    env_name: Optional[str] = None,
    batch_size: int = 256,
    epochs: int = 1,
    shuffle: bool = True,
) -> List[Dict[str, float]]:
    if dataset_path is None and env_name is None:
        raise ValueError("run_training requires either dataset_path or env_name")

    if dataset_path is not None:
        data = load_offline_dataset(dataset_path)
    else:
        data = load_d4rl_dataset(env_name)

    dataset = OfflineReplayDataset(data)
    metrics_history: List[Dict[str, float]] = []
    for _ in range(epochs):
        loader = build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)
        epoch_metrics = trainer.train_sigma_epoch(loader)
        metrics_history.append(epoch_metrics)
    return metrics_history


def main() -> None:
    parser = build_parser()
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument("--env-name", dest="env_name", type=str, default=None)
    parser.add_argument("--state-dim", dest="state_dim", type=int, required=True)
    parser.add_argument("--action-dim", dest="action_dim", type=int, required=True)
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=32)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--epochs", dest="epochs", type=int, default=1)
    parser.add_argument("--sigma-lr", dest="sigma_lr", type=float, default=1e-3)
    args = parser.parse_args()

    trainer = build_trainer_from_args(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lambda_q=args.lambda_q,
        lambda_state_anchor=args.lambda_state_anchor,
        lambda_latent_anchor=args.lambda_latent_anchor,
        q_clip_min=args.q_clip_min,
        q_clip_max=args.q_clip_max,
        sigma_warmup_steps=args.sigma_warmup_steps,
        sigma_lr=args.sigma_lr,
        kats_checkpoint=args.kats_checkpoint,
        critic_checkpoint=args.critic_checkpoint,
    )
    metrics = run_training(
        trainer=trainer,
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        env_name=args.env_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    for epoch, epoch_metrics in enumerate(metrics, start=1):
        print(f"Epoch {epoch}: {epoch_metrics}")


if __name__ == "__main__":
    main()
