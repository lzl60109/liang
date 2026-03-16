from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from vgks.models import ConservativeCritic, KoopmanDynamicsModel


PathLike = Union[str, Path]


def load_kats_checkpoint(model: KoopmanDynamicsModel, checkpoint_path: PathLike) -> KoopmanDynamicsModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_kats_state_dict(checkpoint)
    return model


def load_tgcvg_critic_checkpoint(
    critic: ConservativeCritic, checkpoint_path: PathLike
) -> ConservativeCritic:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    critic.load_tgcvg_state_dict(checkpoint)
    return critic
