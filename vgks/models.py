from __future__ import annotations

import torch
from torch import nn


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class KoopmanDynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = _build_mlp(state_dim, hidden_dim, latent_dim)
        self.decoder = _build_mlp(latent_dim, hidden_dim, state_dim)
        self.layerK = nn.Linear(latent_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.layerK.weight.copy_(0.95 * torch.eye(latent_dim))

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(states.float()))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents.float())

    def predict_next_latent(self, latents: torch.Tensor) -> torch.Tensor:
        return self.layerK(latents.float())


class InverseDynamicsModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.model = _build_mlp(latent_dim * 2, hidden_dim, action_dim)

    def forward(self, latents: torch.Tensor, next_latents: torch.Tensor) -> torch.Tensor:
        features = torch.cat([latents.float(), next_latents.float()], dim=1)
        return self.model(features)


class SigmaModel(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.sigma_layer = nn.Linear(latent_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.sigma_layer.weight.copy_(torch.eye(latent_dim))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.sigma_layer(latents.float())


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = _build_mlp(state_dim + action_dim, hidden_dim, 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observations.float(), actions.float()], dim=1)
        return self.model(inputs).reshape(-1)


class ConservativeCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.q1_net = _QNetwork(state_dim, action_dim, hidden_dim)
        self.q2_net = _QNetwork(state_dim, action_dim, hidden_dim)

    def q1(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q1_net(observations, actions)

    def q2(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q2_net(observations, actions)

    def conservative_value(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.minimum(self.q1(observations, actions), self.q2(observations, actions))
