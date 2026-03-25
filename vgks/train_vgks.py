from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
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
from vgks.models import ConservativeCritic, InverseDynamicsModel, KoopmanDynamicsModel, SigmaModel
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


def infer_dims_from_dataset_source(dataset_path: Path) -> Dict[str, int]:
    data = load_offline_dataset(dataset_path)
    observations = np.asarray(data["observations"])
    actions = np.asarray(data["actions"])
    state_dim = int(observations.shape[-1]) if observations.ndim > 1 else 1
    action_dim = int(actions.shape[-1]) if actions.ndim > 1 else 1
    return {"state_dim": state_dim, "action_dim": action_dim}


def merge_config_with_args(config: Dict[str, object], arg_values: Dict[str, object]) -> Dict[str, object]:
    merged = dict(config)
    for key, value in arg_values.items():
        if value is not None:
            merged[key] = value
    return merged


def _ensure_transition_targets(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normalized = dict(data)
    size = int(normalized["observations"].shape[0])
    if normalized.get("rewards") is None:
        normalized["rewards"] = np.zeros(size, dtype=np.float32)
    if normalized.get("terminals") is None:
        normalized["terminals"] = np.zeros(size, dtype=np.float32)
    return normalized


def _make_eval_env(env_name: Optional[str], state_dim: int, action_dim: int):
    if env_name is None:
        return ToyEvalEnv(state_dim, action_dim)
    return make_env(env_name)


def _soft_update_module(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def _policy_action(policy: BCPolicy, observations: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.tanh(policy(observations)), min=-1.0, max=1.0)


def _sample_policy_actions(
    policy: BCPolicy,
    observations: torch.Tensor,
    num_samples: int,
    noise_std: float = 0.1,
) -> torch.Tensor:
    base_actions = _policy_action(policy, observations)
    repeated = base_actions.unsqueeze(1).expand(-1, num_samples, -1)
    noisy = repeated + noise_std * torch.randn_like(repeated)
    return torch.clamp(noisy, min=-1.0, max=1.0)


def _train_behavior_policy_epoch(
    policy: BCPolicy,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    policy.train()
    total_loss = 0.0
    step_count = 0
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = torch.clamp(batch["actions"].to(device), min=-1.0, max=1.0)
        predicted = _policy_action(policy, observations)
        loss = F.mse_loss(predicted, actions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        step_count += 1
    return {"behavior_loss": total_loss / max(1, step_count), "step_count": step_count}


def _train_koopman_epoch(
    dynamics: KoopmanDynamicsModel,
    inverse_model: InverseDynamicsModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    dynamics.train()
    inverse_model.train()
    for param in dynamics.parameters():
        param.requires_grad_(True)
    for param in inverse_model.parameters():
        param.requires_grad_(True)
    totals = {"state_loss": 0.0, "action_loss": 0.0, "latent_loss": 0.0, "total_loss": 0.0, "step_count": 0}
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        next_observations = batch["next_observations"].to(device)

        z_t = dynamics.encode(observations)
        z_t1 = dynamics.encode(next_observations)
        predicted_next_latent = dynamics.predict_next_latent(z_t)
        predicted_next_state = dynamics.decode(predicted_next_latent)
        reconstructed_actions = inverse_model(z_t, z_t1)

        state_loss = F.mse_loss(predicted_next_state, next_observations)
        action_loss = F.mse_loss(reconstructed_actions, actions)
        latent_loss = F.mse_loss(predicted_next_latent, z_t1.detach())
        total_loss = state_loss + action_loss + 0.1 * latent_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        totals["state_loss"] += float(state_loss.detach().cpu().item())
        totals["action_loss"] += float(action_loss.detach().cpu().item())
        totals["latent_loss"] += float(latent_loss.detach().cpu().item())
        totals["total_loss"] += float(total_loss.detach().cpu().item())
        totals["step_count"] += 1

    steps = max(1, totals["step_count"])
    for key in ("state_loss", "action_loss", "latent_loss", "total_loss"):
        totals[key] /= steps
    return totals


def _train_conservative_critic_epoch(
    critic: ConservativeCritic,
    target_critic: ConservativeCritic,
    behavior_policy: BCPolicy,
    loader,
    critic_optimizer: torch.optim.Optimizer,
    behavior_optimizer: torch.optim.Optimizer,
    device: str,
    discount: float,
    cql_alpha: float,
    tau: float,
    num_action_samples: int,
    cql_temp: float,
) -> Dict[str, float]:
    behavior_metrics = _train_behavior_policy_epoch(behavior_policy, loader, behavior_optimizer, device)
    critic.train()
    target_critic.eval()
    for param in critic.parameters():
        param.requires_grad_(True)
    totals = {
        "critic_loss": 0.0,
        "bellman_loss": 0.0,
        "cql_loss": 0.0,
        "q_mean": 0.0,
        "target_q_mean": 0.0,
        "data_q_mean": 0.0,
        "ood_q_mean": 0.0,
        "cql_gap": 0.0,
        "step_count": 0,
    }
    for batch in loader:
        observations = batch["observations"].to(device)
        actions = torch.clamp(batch["actions"].to(device), min=-1.0, max=1.0)
        next_observations = batch["next_observations"].to(device)
        rewards = batch["rewards"].to(device)
        terminals = batch["terminals"].to(device)

        with torch.no_grad():
            next_actions = _policy_action(behavior_policy, next_observations)
            target_q = rewards + (1.0 - terminals) * discount * target_critic.conservative_value(
                next_observations, next_actions
            )

        q1 = critic.q1(observations, actions)
        q2 = critic.q2(observations, actions)
        bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        random_actions = torch.empty(
            observations.shape[0], num_action_samples, actions.shape[-1], device=device
        ).uniform_(-1.0, 1.0)
        current_policy_actions = _sample_policy_actions(
            behavior_policy, observations, num_samples=num_action_samples
        )
        next_policy_actions = _sample_policy_actions(
            behavior_policy, next_observations, num_samples=num_action_samples
        )

        q1_rand = critic.q1(observations, random_actions)
        q2_rand = critic.q2(observations, random_actions)
        q1_current = critic.q1(observations, current_policy_actions)
        q2_current = critic.q2(observations, current_policy_actions)
        q1_next = critic.q1(observations, next_policy_actions)
        q2_next = critic.q2(observations, next_policy_actions)

        q1_ood = torch.logsumexp(
            torch.cat([q1_rand, q1_current, q1_next], dim=1) / cql_temp,
            dim=1,
        ) * cql_temp
        q2_ood = torch.logsumexp(
            torch.cat([q2_rand, q2_current, q2_next], dim=1) / cql_temp,
            dim=1,
        ) * cql_temp
        cql_gap1 = q1_ood - q1
        cql_gap2 = q2_ood - q2
        cql_loss = cql_alpha * (cql_gap1.mean() + cql_gap2.mean())
        critic_loss = bellman_loss + cql_loss

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
        critic_optimizer.step()
        _soft_update_module(target_critic, critic, tau)

        totals["critic_loss"] += float(critic_loss.detach().cpu().item())
        totals["bellman_loss"] += float(bellman_loss.detach().cpu().item())
        totals["cql_loss"] += float(cql_loss.detach().cpu().item())
        data_q_mean = torch.minimum(q1, q2).mean()
        ood_q_mean = torch.minimum(q1_ood, q2_ood).mean()
        target_q_mean = target_q.mean()
        totals["q_mean"] += float(data_q_mean.detach().cpu().item())
        totals["data_q_mean"] += float(data_q_mean.detach().cpu().item())
        totals["ood_q_mean"] += float(ood_q_mean.detach().cpu().item())
        totals["target_q_mean"] += float(target_q_mean.detach().cpu().item())
        totals["cql_gap"] += float(((cql_gap1.mean() + cql_gap2.mean()) * 0.5).detach().cpu().item())
        totals["step_count"] += 1

    steps = max(1, totals["step_count"])
    for key in ("critic_loss", "bellman_loss", "cql_loss", "q_mean", "target_q_mean", "data_q_mean", "ood_q_mean", "cql_gap"):
        totals[key] /= steps
    totals["behavior_loss"] = behavior_metrics["behavior_loss"]
    return totals


def _build_checkpoint_paths(save_dir: Path) -> Dict[str, Path]:
    return {
        "kats": save_dir / "kats_checkpoint.pt",
        "critic": save_dir / "critic_checkpoint.pt",
        "vgks": save_dir / "vgks_checkpoint.pt",
    }


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
    lambda_action_anchor: float = 0.5,
    q_delta: Optional[float] = None,
    max_state_shift: Optional[float] = None,
    max_action_deviation: Optional[float] = None,
    commute_horizon: int = 1,
    value_temperature: float = 1.0,
    sigma_tau: float = 0.01,
    device: str = "cpu",
) -> ValueGuidedKoopmanTrainer:
    dynamics = KoopmanDynamicsModel(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    inverse_model = InverseDynamicsModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=max(8, hidden_dim))
    sigma_model = SigmaModel(latent_dim=latent_dim)
    critic = ConservativeCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=max(256, hidden_dim))

    if kats_checkpoint:
        load_kats_checkpoint(dynamics, kats_checkpoint, inverse_model=inverse_model)
    if critic_checkpoint:
        load_tgcvg_critic_checkpoint(critic, critic_checkpoint)

    return ValueGuidedKoopmanTrainer(
        dynamics=dynamics,
        sigma_model=sigma_model,
        critic=critic,
        inverse_model=inverse_model,
        action_dim=action_dim,
        sigma_tau=sigma_tau,
        lambda_q=lambda_q,
        lambda_state_anchor=lambda_state_anchor,
        lambda_action_anchor=lambda_action_anchor,
        lambda_latent_anchor=lambda_latent_anchor,
        q_clip_min=q_clip_min,
        q_clip_max=q_clip_max,
        sigma_warmup_steps=sigma_warmup_steps,
        q_delta=q_delta,
        max_state_shift=max_state_shift,
        max_action_deviation=max_action_deviation,
        commute_horizon=commute_horizon,
        value_temperature=value_temperature,
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
    koopman_lr: float = 1e-3,
    inverse_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    critic_policy_lr: float = 3e-4,
    critic_discount: float = 0.99,
    critic_tau: float = 5e-3,
    cql_alpha: float = 1.0,
    critic_samples: int = 10,
    cql_temp: float = 1.0,
) -> Dict[str, object]:
    if dataset_path is None and env_name is None:
        raise ValueError("run_training requires either dataset_path or env_name")

    torch.manual_seed(seed)
    np.random.seed(seed)
    raw_data = load_offline_dataset(dataset_path) if dataset_path is not None else load_d4rl_dataset(env_name)
    data = _ensure_transition_targets(raw_data)
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

    koopman_optimizer = torch.optim.Adam(
        [
            {"params": trainer.dynamics.parameters(), "lr": koopman_lr},
            {"params": trainer.inverse_model.parameters(), "lr": inverse_lr},
        ]
    )
    critic_optimizer = torch.optim.Adam(trainer.critic.parameters(), lr=critic_lr)
    target_critic = deepcopy(trainer.critic).to(device)
    for param in target_critic.parameters():
        param.requires_grad_(False)
    behavior_policy = BCPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    behavior_optimizer = torch.optim.Adam(behavior_policy.parameters(), lr=critic_policy_lr)

    koopman_history: List[Dict[str, float]] = []
    critic_history: List[Dict[str, float]] = []
    sigma_history: List[Dict[str, float]] = []
    eval_history: List[Dict[str, float]] = []
    best_eval: Optional[Dict[str, float]] = None
    best_normalized_score = float("-inf")
    policy = BCPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy_metrics: Dict[str, float] = {}

    for epoch in range(epochs):
        train_loader = build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        koopman_metrics = _train_koopman_epoch(
            trainer.dynamics, trainer.inverse_model, train_loader, koopman_optimizer, device
        )
        koopman_history.append(koopman_metrics)

        critic_loader = build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        critic_metrics = _train_conservative_critic_epoch(
            trainer.critic,
            target_critic,
            behavior_policy,
            critic_loader,
            critic_optimizer,
            behavior_optimizer,
            device,
            discount=critic_discount,
            cql_alpha=cql_alpha,
            tau=critic_tau,
            num_action_samples=critic_samples,
            cql_temp=cql_temp,
        )
        critic_history.append(critic_metrics)

        # Freeze updated modules before sigma optimization.
        trainer._freeze_module(trainer.dynamics)
        trainer._freeze_module(trainer.critic)
        trainer._freeze_module(trainer.inverse_model)

        sigma_loader = build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        sigma_metrics = trainer.train_sigma_epoch(sigma_loader)
        sigma_history.append(sigma_metrics)

        if logger is not None:
            logger.log_metrics({f"koopman/{k}": v for k, v in koopman_metrics.items() if k != "step_count"}, step=epoch + 1)
            logger.log_metrics({f"critic/{k}": v for k, v in critic_metrics.items() if k != "step_count"}, step=epoch + 1)
            logger.log_metrics({f"sigma/{k}": v for k, v in sigma_metrics.items() if k != "step_count"}, step=epoch + 1)

        final_loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        augmented_batches = [trainer.augment_batch(batch) for batch in final_loader]
        augmented = {
            key: torch.cat([batch[key] for batch in augmented_batches], dim=0) if isinstance(augmented_batches[0][key], torch.Tensor) else sum(int(batch[key]) for batch in augmented_batches)
            for key in augmented_batches[0].keys()
        }

        combined = {
            "observations": torch.cat([torch.tensor(data["observations"], dtype=torch.float32), augmented["observations"]], dim=0),
            "actions": torch.cat([torch.tensor(data["actions"], dtype=torch.float32), augmented["actions"]], dim=0),
            "next_observations": torch.cat([torch.tensor(data["next_observations"], dtype=torch.float32), augmented["next_observations"]], dim=0),
        }
        combined_loader = build_dataloader(
            OfflineReplayDataset({key: value.numpy() for key, value in combined.items()}),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        policy_metrics = train_bc_epoch(
            policy,
            combined_loader,
            policy_optimizer,
            device,
            epoch=epoch + 1,
            total_epochs=epochs,
        )
        if logger is not None:
            logger.log_metrics({f"policy/{k}": v for k, v in policy_metrics.items()}, step=epoch + 1)

        if (epoch + 1) % max(1, eval_interval) == 0 or epoch == epochs - 1:
            eval_env = _make_eval_env(env_name, state_dim, action_dim)
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

            if logger is not None:
                logger.log_metrics(
                    {
                        "eval/raw_return": eval_metrics["raw_return"],
                        "eval/normalized_score": eval_metrics["normalized_score"],
                        "eval/best_normalized_score": best_normalized_score,
                    },
                    step=epoch + 1,
                )

    checkpoint_paths = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_paths = _build_checkpoint_paths(save_dir)
        with (save_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({"koopman": koopman_history, "critic": critic_history, "sigma": sigma_history}, handle, indent=2)
        with (save_dir / "eval_history.json").open("w", encoding="utf-8") as handle:
            json.dump(eval_history, handle, indent=2)
        export_augmented_dataset(save_dir / "augmented_dataset.npz", augmented)
        torch.save(
            {"kats": {"dynamics": trainer.dynamics.state_dict(), "inverse_model": trainer.inverse_model.state_dict(), "sigma_model": trainer.sigma_model.state_dict()}},
            checkpoint_paths["kats"],
        )
        torch.save(
            {
                "critic_checkpoint": {
                    "critic": trainer.critic.state_dict(),
                    "critic1_target": target_critic.q1_net.state_dict(),
                    "critic2_target": target_critic.q2_net.state_dict(),
                    "behavior_policy": behavior_policy.state_dict(),
                }
            },
            checkpoint_paths["critic"],
        )
        torch.save(
            {
                "kats_checkpoint": str(checkpoint_paths["kats"]),
                "critic_checkpoint": str(checkpoint_paths["critic"]),
                "sigma_model": trainer.sigma_model.state_dict(),
                "policy": policy.state_dict(),
            },
            checkpoint_paths["vgks"],
        )
        torch.save({"sigma_model": trainer.sigma_model.state_dict(), "policy": policy.state_dict()}, save_dir / "last_checkpoint.pt")
        torch.save({"sigma_model": trainer.sigma_model.state_dict(), "policy": policy.state_dict()}, save_dir / "checkpoint.pt")
        if logger is not None:
            final_eval = eval_history[-1] if eval_history else {"raw_return": 0.0, "normalized_score": 0.0, "episodes": eval_episodes}
            logger.write_eval(final_eval)
            logger.finish()

    checkpoints = None if checkpoint_paths is None else {key: str(value) for key, value in checkpoint_paths.items()}
    return {
        "koopman_history": koopman_history,
        "critic_history": critic_history,
        "sigma_history": sigma_history,
        "eval_history": eval_history,
        "best_normalized_score": best_normalized_score,
        "best_eval": best_eval,
        "checkpoints": checkpoints,
        "last": {
            "koopman": koopman_history[-1] if koopman_history else None,
            "critic": critic_history[-1] if critic_history else None,
            "sigma": sigma_history[-1] if sigma_history else None,
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
    parser.add_argument("--koopman-lr", dest="koopman_lr", type=float, default=None)
    parser.add_argument("--inverse-lr", dest="inverse_lr", type=float, default=None)
    parser.add_argument("--critic-lr", dest="critic_lr", type=float, default=None)
    parser.add_argument("--critic-policy-lr", dest="critic_policy_lr", type=float, default=None)
    parser.add_argument("--critic-discount", dest="critic_discount", type=float, default=None)
    parser.add_argument("--critic-tau", dest="critic_tau", type=float, default=None)
    parser.add_argument("--critic-samples", dest="critic_samples", type=int, default=None)
    parser.add_argument("--cql-alpha", dest="cql_alpha", type=float, default=None)
    parser.add_argument("--cql-temp", dest="cql_temp", type=float, default=None)
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
        sigma_tau=None,
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
        koopman_lr=None,
        inverse_lr=None,
        critic_lr=None,
        critic_policy_lr=None,
        critic_discount=None,
        critic_tau=None,
        critic_samples=None,
        cql_alpha=None,
        cql_temp=None,
        device=None,
        num_workers=None,
        lambda_q=None,
        lambda_state_anchor=None,
        lambda_action_anchor=None,
        lambda_latent_anchor=None,
        q_clip_min=None,
        q_clip_max=None,
        q_threshold=None,
        q_delta=None,
        max_state_shift=None,
        max_action_deviation=None,
        commute_horizon=None,
        value_temperature=None,
        sigma_warmup_steps=None,
        kats_checkpoint=None,
        critic_checkpoint=None,
    )
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

    trainer = build_trainer_from_args(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=merged["latent_dim"],
        hidden_dim=merged["hidden_dim"],
        sigma_tau=merged["sigma_tau"],
        lambda_q=merged["lambda_q"],
        lambda_state_anchor=merged["lambda_state_anchor"],
        lambda_action_anchor=merged["lambda_action_anchor"],
        lambda_latent_anchor=merged["lambda_latent_anchor"],
        q_clip_min=merged["q_clip_min"],
        q_clip_max=merged["q_clip_max"],
        sigma_warmup_steps=merged["sigma_warmup_steps"],
        sigma_lr=merged["sigma_lr"],
        kats_checkpoint=merged.get("kats_checkpoint"),
        critic_checkpoint=merged.get("critic_checkpoint"),
        q_delta=merged.get("q_delta", 0.0),
        max_state_shift=merged.get("max_state_shift"),
        max_action_deviation=merged.get("max_action_deviation"),
        commute_horizon=merged.get("commute_horizon", 1),
        value_temperature=merged.get("value_temperature", 1.0),
        device=merged["device"],
    )
    metrics = run_training(
        trainer=trainer,
        dataset_path=dataset_path,
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
        koopman_lr=merged["koopman_lr"],
        inverse_lr=merged["inverse_lr"],
        critic_lr=merged["critic_lr"],
        critic_policy_lr=merged["critic_policy_lr"],
        critic_discount=merged["critic_discount"],
        critic_tau=merged["critic_tau"],
        cql_alpha=merged["cql_alpha"],
        critic_samples=merged["critic_samples"],
        cql_temp=merged["cql_temp"],
    )
    print(json.dumps(metrics["last"], indent=2))


if __name__ == "__main__":
    main()
