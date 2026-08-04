from __future__ import annotations

import torch.nn as nn


def count_model_parameters(model: nn.Module) -> dict:
    """Count total, trainable, and frozen parameters."""
    total_params = sum(parameter.numel() for parameter in model.parameters())

    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    frozen_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }


def print_model_parameters(
    model: nn.Module,
    model_name: str = "Model",
) -> None:
    """Print model parameter summary."""
    param_info = count_model_parameters(model)

    print("=" * 80)
    print(f"[{model_name}] Parameter Summary")
    print(f"Total parameters:     {param_info['total_params']:,}")
    print(f"Trainable parameters: {param_info['trainable_params']:,}")
    print(f"Frozen parameters:    {param_info['frozen_params']:,}")
    print("=" * 80)