from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observations.float(), actions.float()], dim=1)
        return self.network(inputs).reshape(-1)


class KoopmanDynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, latent_dim)
        self.layerK = nn.Linear(latent_dim, latent_dim, bias=False)

        self.layer3inv = nn.Linear(latent_dim, hidden_dim)
        self.layer2inv = nn.Linear(hidden_dim, hidden_dim)
        self.layer1inv = nn.Linear(hidden_dim, state_dim)

        with torch.no_grad():
            self.layerK.weight.copy_(0.95 * torch.eye(latent_dim))

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.layer1(states.float()))
        x = torch.tanh(self.layer2(x))
        return torch.tanh(self.layer3(x))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.layer3inv(latents.float()))
        x = torch.tanh(self.layer2inv(x))
        return self.layer1inv(x)

    def predict_next_latent(self, latents: torch.Tensor) -> torch.Tensor:
        return self.layerK(latents.float())

    def load_kats_state_dict(self, checkpoint: Dict[str, torch.Tensor]) -> None:
        mapped = {
            "layer1.weight": checkpoint["layer1.weight"],
            "layer1.bias": checkpoint["layer1.bias"],
            "layer2.weight": checkpoint["layer2.weight"],
            "layer2.bias": checkpoint["layer2.bias"],
            "layer3.weight": checkpoint["layer3.weight"],
            "layer3.bias": checkpoint["layer3.bias"],
            "layerK.weight": checkpoint["layerK.weight"],
            "layer3inv.weight": checkpoint["layer3inv.weight"],
            "layer3inv.bias": checkpoint["layer3inv.bias"],
            "layer2inv.weight": checkpoint["layer2inv.weight"],
            "layer2inv.bias": checkpoint["layer2inv.bias"],
            "layer1inv.weight": checkpoint["layer1inv.weight"],
            "layer1inv.bias": checkpoint["layer1inv.bias"],
        }
        self.load_state_dict(mapped, strict=True)


class InverseDynamicsModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, latents: torch.Tensor, next_latents: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latents.float(), next_latents.float()], dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class SigmaModel(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.sigma_layer = nn.Linear(latent_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.sigma_layer.weight.copy_(torch.eye(latent_dim))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.sigma_layer(latents.float())


class ConservativeCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.q1_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2_net = QNetwork(state_dim, action_dim, hidden_dim)

    def q1(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q1_net(observations, actions)

    def q2(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q2_net(observations, actions)

    def conservative_value(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.minimum(self.q1(observations, actions), self.q2(observations, actions))

    def load_tgcvg_state_dict(self, checkpoint: Dict[str, Dict[str, torch.Tensor]]) -> None:
        critic_state = checkpoint.get("critic") if isinstance(checkpoint, dict) else None
        if critic_state is not None:
            self.load_state_dict(critic_state, strict=True)
            return
        q1_state = checkpoint.get("critic1") or checkpoint.get("critic_1")
        q2_state = checkpoint.get("critic2") or checkpoint.get("critic_2")
        if q1_state is None or q2_state is None:
            raise KeyError("Expected checkpoint to contain critic1/critic2 or critic_1/critic_2")
        self.q1_net.load_state_dict(q1_state, strict=True)
        self.q2_net.load_state_dict(q2_state, strict=True)
