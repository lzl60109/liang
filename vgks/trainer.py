from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from vgks.models import ConservativeCritic, InverseDynamicsModel, KoopmanDynamicsModel, SigmaModel


@dataclass
class SigmaLossMetrics:
    total_loss: float
    commutation_loss: float
    value_loss: float
    state_anchor_loss: float
    latent_anchor_loss: float
    mean_conservative_q: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_loss": self.total_loss,
            "commutation_loss": self.commutation_loss,
            "value_loss": self.value_loss,
            "state_anchor_loss": self.state_anchor_loss,
            "latent_anchor_loss": self.latent_anchor_loss,
            "mean_conservative_q": self.mean_conservative_q,
        }


class ValueGuidedKoopmanTrainer:
    def __init__(
        self,
        dynamics: KoopmanDynamicsModel,
        sigma_model: SigmaModel,
        critic: ConservativeCritic,
        action_dim: int,
        inverse_model: Optional[InverseDynamicsModel] = None,
        sigma_tau: float = 1.0,
        lambda_q: float = 0.1,
        lambda_state_anchor: float = 1.0,
        lambda_latent_anchor: float = 0.1,
        q_clip_min: float = -20.0,
        q_clip_max: float = 20.0,
        sigma_warmup_steps: int = 0,
        sigma_lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.dynamics = dynamics.to(self.device)
        self.sigma_model = sigma_model.to(self.device)
        self.critic = critic.to(self.device)
        self.inverse_model = (inverse_model or InverseDynamicsModel(
            latent_dim=dynamics.latent_dim,
            action_dim=action_dim,
            hidden_dim=max(8, dynamics.latent_dim * 2),
        )).to(self.device)
        self.sigma_tau = sigma_tau
        self.lambda_q = lambda_q
        self.lambda_state_anchor = lambda_state_anchor
        self.lambda_latent_anchor = lambda_latent_anchor
        self.q_clip_min = q_clip_min
        self.q_clip_max = q_clip_max
        self.sigma_warmup_steps = sigma_warmup_steps
        self._steps = 0
        self.sigma_optimizer = torch.optim.Adam(self.sigma_model.parameters(), lr=sigma_lr)

        self._freeze_module(self.dynamics)
        self._freeze_module(self.critic)
        self._freeze_module(self.inverse_model)

    def _freeze_module(self, module: torch.nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def _to_tensor(self, values) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(device=self.device, dtype=torch.float32)
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _effective_lambda_q(self) -> float:
        return 0.0 if self._steps < self.sigma_warmup_steps else self.lambda_q

    def compute_sigma_loss_tensors(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        observations = self._to_tensor(batch["observations"])
        next_observations = self._to_tensor(batch["next_observations"])

        z_t = self.dynamics.encode(observations)
        z_t1 = self.dynamics.encode(next_observations)
        k_z_t = self.dynamics.predict_next_latent(z_t)
        error = z_t1 - k_z_t
        weights = torch.exp(self.sigma_tau * torch.sum(error.detach() ** 2, dim=1, keepdim=True))

        sigma_z_t = self.sigma_model(z_t)
        sigma_z_t1 = self.sigma_model(z_t1)

        k_sigma_z_t = self.dynamics.predict_next_latent(sigma_z_t)
        sigma_loss_vec = k_sigma_z_t - sigma_z_t1
        commutation_loss = F.mse_loss(weights * sigma_loss_vec, torch.zeros_like(sigma_loss_vec))

        augmented_states = self.dynamics.decode(sigma_z_t)
        augmented_next_states = self.dynamics.decode(sigma_z_t1)
        augmented_actions = self.inverse_model(sigma_z_t, sigma_z_t1)

        conservative_q = self.critic.conservative_value(augmented_states, augmented_actions)
        conservative_q = torch.clamp(conservative_q, min=self.q_clip_min, max=self.q_clip_max)
        value_loss = -conservative_q.mean()

        state_anchor_loss = (
            F.mse_loss(augmented_states, observations)
            + F.mse_loss(augmented_next_states, next_observations)
        )
        latent_anchor_loss = F.mse_loss(sigma_z_t, z_t) + F.mse_loss(sigma_z_t1, z_t1)

        total_loss = (
            commutation_loss
            + self._effective_lambda_q() * value_loss
            + self.lambda_state_anchor * state_anchor_loss
            + self.lambda_latent_anchor * latent_anchor_loss
        )

        return {
            "total_loss": total_loss,
            "commutation_loss": commutation_loss,
            "value_loss": value_loss,
            "state_anchor_loss": state_anchor_loss,
            "latent_anchor_loss": latent_anchor_loss,
            "mean_conservative_q": conservative_q.mean(),
        }

    def compute_sigma_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        tensor_metrics = self.compute_sigma_loss_tensors(batch)
        self._steps += 1
        return SigmaLossMetrics(
            total_loss=float(tensor_metrics["total_loss"].detach().cpu().item()),
            commutation_loss=float(tensor_metrics["commutation_loss"].detach().cpu().item()),
            value_loss=float(tensor_metrics["value_loss"].detach().cpu().item()),
            state_anchor_loss=float(tensor_metrics["state_anchor_loss"].detach().cpu().item()),
            latent_anchor_loss=float(tensor_metrics["latent_anchor_loss"].detach().cpu().item()),
            mean_conservative_q=float(tensor_metrics["mean_conservative_q"].detach().cpu().item()),
        ).to_dict()

    def train_sigma_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.sigma_model.train()
        tensor_metrics = self.compute_sigma_loss_tensors(batch)
        self.sigma_optimizer.zero_grad()
        tensor_metrics["total_loss"].backward()
        self.sigma_optimizer.step()
        self._steps += 1
        return {
            key: float(value.detach().cpu().item()) for key, value in tensor_metrics.items()
        }

    def train_sigma_epoch(self, batches: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        totals = {
            "total_loss": 0.0,
            "commutation_loss": 0.0,
            "value_loss": 0.0,
            "state_anchor_loss": 0.0,
            "latent_anchor_loss": 0.0,
            "mean_conservative_q": 0.0,
        }
        step_count = 0
        for batch in batches:
            step_metrics = self.train_sigma_step(batch)
            step_count += 1
            for key in totals:
                totals[key] += step_metrics[key]

        if step_count == 0:
            raise ValueError("train_sigma_epoch requires at least one batch")

        averaged = {key: value / step_count for key, value in totals.items()}
        averaged["step_count"] = step_count
        return averaged

    def augment_batch(
        self, batch: Dict[str, torch.Tensor], q_threshold: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            observations = self._to_tensor(batch["observations"])
            next_observations = self._to_tensor(batch["next_observations"])

            z_t = self.dynamics.encode(observations)
            z_t1 = self.dynamics.encode(next_observations)
            sigma_z_t = self.sigma_model(z_t)
            sigma_z_t1 = self.sigma_model(z_t1)

            augmented_states = self.dynamics.decode(sigma_z_t)
            augmented_next_states = self.dynamics.decode(sigma_z_t1)
            augmented_actions = self.inverse_model(sigma_z_t, sigma_z_t1)
            q_values = torch.clamp(
                self.critic.conservative_value(augmented_states, augmented_actions),
                min=self.q_clip_min,
                max=self.q_clip_max,
            )

            if q_threshold is None:
                mask = torch.ones_like(q_values, dtype=torch.bool)
            else:
                mask = q_values >= q_threshold

            return {
                "observations": augmented_states[mask].detach().cpu(),
                "actions": augmented_actions[mask].detach().cpu(),
                "next_observations": augmented_next_states[mask].detach().cpu(),
                "q_values": q_values[mask].detach().cpu(),
                "num_kept": int(mask.sum().detach().cpu().item()),
            }
