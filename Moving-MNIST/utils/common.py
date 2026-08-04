from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(
    seed: int | None,
    deterministic: bool = False,
) -> None:
    """Fix random seed.

    Set seed to -1 or None if you do not want deterministic behavior.
    """
    if seed is None or seed < 0:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device: str = "auto") -> torch.device:
    """Choose CPU or CUDA device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path