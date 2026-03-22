from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from vgks.models import ConservativeCritic, InverseDynamicsModel, KoopmanDynamicsModel


PathLike = Union[str, Path]


def load_kats_checkpoint(
    model: KoopmanDynamicsModel,
    checkpoint_path: PathLike,
    *,
    inverse_model: Optional[InverseDynamicsModel] = None,
) -> KoopmanDynamicsModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if "kats" in checkpoint:
        checkpoint = checkpoint["kats"]

    dynamics_state = checkpoint.get("dynamics") if isinstance(checkpoint, dict) else None
    if dynamics_state is not None:
        model.load_state_dict(dynamics_state, strict=True)
    else:
        model.load_kats_state_dict(checkpoint)

    if inverse_model is not None and isinstance(checkpoint, dict):
        inverse_state = checkpoint.get("inverse_model")
        if inverse_state is not None:
            inverse_model.load_state_dict(inverse_state, strict=True)
    return model


def load_tgcvg_critic_checkpoint(
    critic: ConservativeCritic, checkpoint_path: PathLike
) -> ConservativeCritic:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "critic_checkpoint" in checkpoint:
        checkpoint = checkpoint["critic_checkpoint"]
    critic.load_tgcvg_state_dict(checkpoint)
    return critic
