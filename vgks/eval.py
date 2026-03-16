from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    return result


@torch.no_grad()
def evaluate_policy(env, policy, device: str = "cpu", n_episodes: int = 10) -> Dict[str, float]:
    policy.eval()
    returns = []
    for _ in range(n_episodes):
        observation = _reset_env(env)
        done = False
        episode_return = 0.0
        while not done:
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy(obs_tensor)
            if isinstance(action, tuple):
                action = action[0]
            action_np = action.squeeze(0).detach().cpu().numpy()
            observation, reward, done, _ = _step_env(env, action_np)
            episode_return += float(reward)
        returns.append(episode_return)

    raw_return = float(np.mean(returns))
    normalized_score = float(env.get_normalized_score(raw_return) * 100.0)
    return {
        "raw_return": raw_return,
        "normalized_score": normalized_score,
        "episodes": int(n_episodes),
    }
