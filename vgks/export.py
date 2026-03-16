from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch


def export_augmented_dataset(path: Path, augmented: Dict[str, torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, value in augmented.items():
        if isinstance(value, torch.Tensor):
            arrays[key] = value.detach().cpu().numpy()
        else:
            arrays[key] = np.asarray(value)
    np.savez(path, **arrays)
