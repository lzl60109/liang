from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


@dataclass
class LinearLayer:
    in_features: int
    out_features: int
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / max(1, self.in_features)
        self.weight = rng.normal(0.0, scale, size=(self.in_features, self.out_features)).astype(
            np.float32
        )
        self.bias = np.zeros(self.out_features, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias


class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int) -> None:
        self.fc1 = LinearLayer(input_dim, hidden_dim, seed)
        self.fc2 = LinearLayer(hidden_dim, hidden_dim, seed + 1)
        self.fc3 = LinearLayer(hidden_dim, output_dim, seed + 2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = _tanh(self.fc1(x))
        x = _tanh(self.fc2(x))
        return self.fc3(x)


class KoopmanDynamicsModel:
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 64) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = MLP(state_dim, hidden_dim, latent_dim, seed=11)
        self.decoder = MLP(latent_dim, hidden_dim, state_dim, seed=19)
        self.layerK = np.eye(latent_dim, dtype=np.float32) * 0.95

    def encode(self, states: np.ndarray) -> np.ndarray:
        return _tanh(self.encoder(np.asarray(states, dtype=np.float32)))

    def decode(self, latents: np.ndarray) -> np.ndarray:
        return self.decoder(np.asarray(latents, dtype=np.float32))

    def predict_next_latent(self, latents: np.ndarray) -> np.ndarray:
        return np.asarray(latents, dtype=np.float32) @ self.layerK.T


class InverseDynamicsModel:
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        self.model = MLP(latent_dim * 2, hidden_dim, action_dim, seed=29)

    def __call__(self, latents: np.ndarray, next_latents: np.ndarray) -> np.ndarray:
        features = np.concatenate([latents, next_latents], axis=1)
        return self.model(features)


class SigmaModel:
    def __init__(self, latent_dim: int) -> None:
        self.weight = np.eye(latent_dim, dtype=np.float32)

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        return np.asarray(latents, dtype=np.float32) @ self.weight.T


class _QNetwork:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, seed: int) -> None:
        self.model = MLP(state_dim + action_dim, hidden_dim, 1, seed)

    def __call__(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        inputs = np.concatenate(
            [np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)],
            axis=1,
        )
        return self.model(inputs).reshape(-1)


class ConservativeCritic:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        self._q1 = _QNetwork(state_dim, action_dim, hidden_dim, seed=41)
        self._q2 = _QNetwork(state_dim, action_dim, hidden_dim, seed=53)

    def q1(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._q1(observations, actions)

    def q2(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._q2(observations, actions)

    def conservative_value(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.minimum(self.q1(observations, actions), self.q2(observations, actions))
