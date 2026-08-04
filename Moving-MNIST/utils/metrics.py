from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mse_value(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def mae_value(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def psnr_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute image PSNR as an evaluation metric."""
    mse = F.mse_loss(pred, target).clamp_min(eps)
    return 20.0 * torch.log10(pred.new_tensor(float(data_range))) - 10.0 * torch.log10(mse)


def _to_nchw_frames(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B,T,H,W,C] or [B,T,C,H,W] image sequences to [B*T,C,H,W]."""
    if tensor.ndim != 5:
        raise ValueError(f"Expected a 5D image sequence tensor, got {tuple(tensor.shape)}.")

    if tensor.shape[-1] <= 4:
        tensor = tensor.permute(0, 1, 4, 2, 3)
    elif tensor.shape[2] <= 4:
        pass
    else:
        raise ValueError(f"Cannot infer image channel axis from shape {tuple(tensor.shape)}.")

    return tensor.reshape(-1, tensor.shape[2], tensor.shape[3], tensor.shape[4])


def _gaussian_kernel(
    channels: int,
    kernel_size: int,
    sigma: float,
    device,
    dtype,
) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()


def _ssim_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
    kernel_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean SSIM and contrast-structure components for NCHW images."""
    _, channels, height, width = pred.shape
    kernel_size = min(kernel_size, height, width)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(kernel_size, 3)
    padding = kernel_size // 2
    kernel = _gaussian_kernel(channels, kernel_size, sigma, pred.device, pred.dtype)

    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = (
        F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_target_sq
    )
    sigma_cross = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_cross

    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    c1 = pred.new_tensor(c1)
    c2 = pred.new_tensor(c2)

    luminance = (2.0 * mu_cross + c1) / (mu_pred_sq + mu_target_sq + c1 + eps)
    contrast_structure = (2.0 * sigma_cross + c2) / (sigma_pred_sq + sigma_target_sq + c2 + eps)
    ssim_map = luminance * contrast_structure

    return ssim_map.flatten(1).mean(dim=1).mean(), contrast_structure.flatten(1).mean(dim=1).mean()


def ms_ssim_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    weights: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a lightweight MS-SSIM evaluation metric for image sequences.

    Inputs may be [B,T,H,W,C] or [B,T,C,H,W]. Values are clamped to the
    valid display range for the metric only; model outputs and losses are
    unaffected.
    """
    pred = _to_nchw_frames(pred.float())
    target = _to_nchw_frames(target.float())
    data_range = max(float(data_range), eps)
    pred = pred.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)

    levels = min(len(weights), max(1, int(math.floor(math.log2(min(pred.shape[-2:])))) - 1))
    levels = max(1, levels)
    active_weights = pred.new_tensor(weights[:levels])
    active_weights = active_weights / active_weights.sum()

    contrast_terms = []
    current_pred = pred
    current_target = target

    for level in range(levels):
        ssim, contrast_structure = _ssim_components(
            current_pred,
            current_target,
            data_range=data_range,
            eps=eps,
        )
        if level < levels - 1:
            contrast_terms.append(contrast_structure.clamp_min(eps))
            current_pred = F.avg_pool2d(current_pred, kernel_size=2, stride=2)
            current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)
        else:
            contrast_terms.append(ssim.clamp_min(eps))

    terms = torch.stack(contrast_terms)
    return torch.prod(terms**active_weights)


def image_quality_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> dict:
    """Return PSNR and MS-SSIM as Python floats for logging."""
    with torch.no_grad():
        return {
            "psnr": float(psnr_value(pred, target, data_range=data_range).detach().cpu()),
            "ms_ssim": float(ms_ssim_value(pred, target, data_range=data_range).detach().cpu()),
        }


def huber_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)


def _relative_reliability_label(
    error: torch.Tensor,
    floor: float = 0.05,
) -> torch.Tensor:
    """Convert error to a sample-local relative reliability label.

    Each sample is normalized across its own future horizon, so the label for
    one sample does not depend on which other samples happen to share a batch.
    """
    mean_error = error.mean(dim=1, keepdim=True)
    std_error = error.std(
        dim=1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-6)
    z_score = (error - mean_error) / std_error

    label = torch.sigmoid(-z_score)
    label = label.clamp(floor, 1.0 - floor)

    return label


def _absolute_reliability_label(
    error: torch.Tensor,
    tau: float,
    floor: float = 0.05,
) -> torch.Tensor:
    label = torch.exp(-error / max(tau, 1e-6))
    label = label.clamp(floor, 1.0 - floor)

    return label


def build_reliability_labels(
    p_img: torch.Tensor,
    y_img: torch.Tensor,
    p_num: torch.Tensor,
    y_num: torch.Tensor,
    tau_img: float = 0.1,
    tau_num: float = 1.0,
    mode: str = "mixed",
    mix: float = 0.5,
    floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build reliability labels from tendency prediction errors.

    mode="absolute":
        Similar to the original design, using exp(-error/tau).

    mode="relative":
        Rank-like reliability inside the current mini-batch. It is less
        sensitive to the absolute scale of image/numerical errors.

    mode="mixed":
        Combine absolute and relative labels. This is the default beta5 setting.
    """
    image_error = ((p_img - y_img) ** 2).mean(dim=(2, 3, 4), keepdim=False)
    image_error = image_error.unsqueeze(-1)

    numerical_error = torch.abs(p_num - y_num).mean(dim=-1)
    numerical_error = numerical_error.unsqueeze(-1)

    img_abs = _absolute_reliability_label(
        error=image_error,
        tau=tau_img,
        floor=floor,
    )

    num_abs = _absolute_reliability_label(
        error=numerical_error,
        tau=tau_num,
        floor=floor,
    )

    img_rel = _relative_reliability_label(
        error=image_error,
        floor=floor,
    )

    num_rel = _relative_reliability_label(
        error=numerical_error,
        floor=floor,
    )

    mode = mode.lower()

    if mode == "absolute":
        r_img_label = img_abs
        r_num_label = num_abs
    elif mode == "relative":
        r_img_label = img_rel
        r_num_label = num_rel
    elif mode == "mixed":
        mix = float(min(max(mix, 0.0), 1.0))
        r_img_label = mix * img_abs + (1.0 - mix) * img_rel
        r_num_label = mix * num_abs + (1.0 - mix) * num_rel
    else:
        raise ValueError(
            "Unknown reliability_label_mode. " "Use 'absolute', 'relative', or 'mixed'."
        )

    r_img_label = r_img_label.clamp(floor, 1.0 - floor)
    r_num_label = r_num_label.clamp(floor, 1.0 - floor)

    return r_img_label.detach(), r_num_label.detach()


def reliability_loss_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    loss_type = loss_type.lower()

    if loss_type == "mse":
        return F.mse_loss(pred, target)

    if loss_type in ["smooth_l1", "huber"]:
        return F.smooth_l1_loss(pred, target)

    if loss_type == "bce":
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(pred, target)

    raise ValueError("Unknown rel_loss_type. Use 'smooth_l1', 'mse', or 'bce'.")


def numerical_loss_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    loss_type = loss_type.lower()

    if loss_type == "mae":
        return mae_value(pred, target)

    if loss_type == "mse":
        return mse_value(pred, target)

    if loss_type == "huber":
        return huber_value(pred, target, delta=huber_delta)

    raise ValueError("Unknown num_loss_type. Use 'mae', 'mse', or 'huber'.")


def image_loss_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Range-agnostic image forecasting loss."""
    loss_type = loss_type.lower()

    if loss_type == "mae":
        return mae_value(pred, target)

    if loss_type == "mse":
        return mse_value(pred, target)

    if loss_type == "huber":
        return huber_value(pred, target, delta=huber_delta)

    raise ValueError("Unknown image_loss_type. Use 'mae', 'mse', or 'huber'.")


def prior_consistency_loss(
    outputs: dict,
    r_img_label: torch.Tensor,
    r_num_label: torch.Tensor,
) -> torch.Tensor:
    """Encourage high-reliability tendencies to remain useful anchors.

    If the tendency label is high, the final prediction should not unnecessarily
    deviate from the tendency. If the tendency label is low, this term becomes
    weak and allows larger correction.
    """
    img_consistency = (
        r_img_label.view(*r_img_label.shape[:2], 1, 1, 1)
        * torch.abs(outputs["y_img"] - outputs["p_img"])
    ).mean()

    num_consistency = (
        r_num_label.expand_as(outputs["y_num"]) * torch.abs(outputs["y_num"] - outputs["p_num"])
    ).mean()

    return img_consistency + num_consistency


def gemininet_loss(
    outputs: dict,
    y_img: torch.Tensor,
    y_num: torch.Tensor,
    lambda_img: float = 1.0,
    lambda_num: float = 1.0,
    lambda_rel: float = 0.02,
    tau_img: float = 0.1,
    tau_num: float = 1.0,
    lambda_img_mae: float = 0.05,
    image_loss_type: str = "mse",
    image_huber_delta: float = 1.0,
    num_loss_type: str = "huber",
    huber_delta: float = 1.0,
    reliability_label_mode: str = "mixed",
    reliability_label_mix: float = 0.5,
    reliability_label_floor: float = 0.05,
    rel_loss_type: str = "smooth_l1",
    lambda_prior_consistency: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    image_mse = mse_value(outputs["y_img"], y_img)
    image_mae = mae_value(outputs["y_img"], y_img)
    image_primary_loss = image_loss_value(
        pred=outputs["y_img"],
        target=y_img,
        loss_type=image_loss_type,
        huber_delta=image_huber_delta,
    )
    image_task_loss = image_primary_loss + lambda_img_mae * image_mae

    num_mse = mse_value(outputs["y_num"], y_num)
    num_mae = mae_value(outputs["y_num"], y_num)
    num_task_loss = numerical_loss_value(
        pred=outputs["y_num"],
        target=y_num,
        loss_type=num_loss_type,
        huber_delta=huber_delta,
    )

    use_reliability_estimator = bool(outputs.get("use_reliability_estimator", True))

    if use_reliability_estimator:
        r_img_label, r_num_label = build_reliability_labels(
            p_img=outputs["p_img"],
            y_img=y_img,
            p_num=outputs["p_num"],
            y_num=y_num,
            tau_img=tau_img,
            tau_num=tau_num,
            mode=reliability_label_mode,
            mix=reliability_label_mix,
            floor=reliability_label_floor,
        )

        rel_img_loss = reliability_loss_value(
            pred=outputs["r_img"],
            target=r_img_label,
            loss_type=rel_loss_type,
        )

        rel_num_loss = reliability_loss_value(
            pred=outputs["r_num"],
            target=r_num_label,
            loss_type=rel_loss_type,
        )

        rel_loss = rel_img_loss + rel_num_loss

        consistency_loss = prior_consistency_loss(
            outputs=outputs,
            r_img_label=r_img_label,
            r_num_label=r_num_label,
        )

        r_img_label_mean = r_img_label.mean()
        r_num_label_mean = r_num_label.mean()

    else:
        rel_img_loss = image_mse.new_tensor(0.0)
        rel_num_loss = image_mse.new_tensor(0.0)
        rel_loss = image_mse.new_tensor(0.0)
        consistency_loss = image_mse.new_tensor(0.0)
        r_img_label_mean = image_mse.new_tensor(1.0)
        r_num_label_mean = image_mse.new_tensor(1.0)

    total_loss = (
        lambda_img * image_task_loss
        + lambda_num * num_task_loss
        + lambda_rel * rel_loss
        + lambda_prior_consistency * consistency_loss
    )

    metrics = {
        "total_loss": float(total_loss.detach().cpu()),
        "image_mse": float(image_mse.detach().cpu()),
        "image_mae": float(image_mae.detach().cpu()),
        "image_task_loss": float(image_task_loss.detach().cpu()),
        "num_mse": float(num_mse.detach().cpu()),
        "num_mae": float(num_mae.detach().cpu()),
        "num_task_loss": float(num_task_loss.detach().cpu()),
        "rel_loss": float(rel_loss.detach().cpu()),
        "rel_img_loss": float(rel_img_loss.detach().cpu()),
        "rel_num_loss": float(rel_num_loss.detach().cpu()),
        "prior_consistency_loss": float(consistency_loss.detach().cpu()),
        "r_img_mean": float(outputs["r_img"].detach().mean().cpu()),
        "r_num_mean": float(outputs["r_num"].detach().mean().cpu()),
        "r_img_label_mean": float(r_img_label_mean.detach().cpu()),
        "r_num_label_mean": float(r_num_label_mean.detach().cpu()),
        "use_reliability_estimator": float(use_reliability_estimator),
    }

    return total_loss, metrics