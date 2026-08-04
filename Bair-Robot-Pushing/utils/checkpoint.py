from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    epoch: int = 0,
    best_metric: float = float("inf"),
    extra: dict | None = None,
) -> None:
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    map_location="cpu",
) -> dict:
    """Load checkpoint.

    weights_only=False is used because our checkpoints contain metadata
    such as dictionaries and training arguments.
    """
    return torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )