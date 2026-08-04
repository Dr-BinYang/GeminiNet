from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure,
        peak_signal_noise_ratio,
    )
except ImportError:
    multiscale_structural_similarity_index_measure = None
    peak_signal_noise_ratio = None


def mse_value(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error."""
    return F.mse_loss(pred, target)


def mae_value(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error."""
    return F.l1_loss(pred, target)


def rmse_value(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Root mean squared error."""
    return torch.sqrt(mse_value(pred, target).clamp_min(0.0))


def _image_sequence_to_nchw(x: torch.Tensor) -> torch.Tensor:
    """Convert [B,T,H,W,C] image sequences to [B*T,C,H,W]."""
    if x.ndim != 5:
        raise ValueError(
            "Image quality metrics expect image sequence tensor with shape "
            f"[B,T,H,W,C], got {tuple(x.shape)}."
        )

    return x.permute(0, 1, 4, 2, 3).reshape(
        x.shape[0] * x.shape[1],
        x.shape[4],
        x.shape[2],
        x.shape[3],
    )


def image_quality_metric_values(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    space: str = "norm",
) -> dict:
    """Compute image PSNR and 3-scale MS-SSIM for reporting only."""
    if peak_signal_noise_ratio is None:
        return {
            f"img_psnr_{space}": float("nan"),
            f"img_ms_ssim_{space}": float("nan"),
        }

    with torch.no_grad():
        pred = _image_sequence_to_nchw(pred.detach().float())
        target = _image_sequence_to_nchw(target.detach().float())

        data_range = max(float(data_range), 1e-6)
        pred = pred.clamp(0.0, data_range)
        target = target.clamp(0.0, data_range)

        psnr = peak_signal_noise_ratio(
            preds=pred,
            target=target,
            data_range=data_range,
        )

        if multiscale_structural_similarity_index_measure is None:
            ms_ssim = psnr.new_tensor(float("nan"))
        else:
            ms_ssim = multiscale_structural_similarity_index_measure(
                preds=pred,
                target=target,
                data_range=data_range,
                kernel_size=3,
                betas=(0.4, 0.3, 0.3),
            )

    return {
        f"img_psnr_{space}": float(psnr.detach().cpu()),
        f"img_ms_ssim_{space}": float(ms_ssim.detach().cpu()),
    }


def huber_value(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 / Huber loss.

    It is less sensitive to occasional large numerical errors than pure MSE,
    and smoother around zero than MAE. This is useful for normalized numerical
    state forecasting.
    """
    return F.smooth_l1_loss(
        pred,
        target,
        beta=beta,
    )


def _absolute_reliability_label(
    error: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Convert an error value into an absolute reliability label."""
    return torch.exp(-error / max(tau, 1e-6)).clamp(0.0, 1.0)


def _relative_reliability_label(error: torch.Tensor) -> torch.Tensor:
    """Convert an error value into a batch-relative reliability label.

    This makes reliability supervision less sensitive to the absolute scale of
    image/numerical normalization. It is helpful when the project is switched
    from [0,1] pixels to z-score physical fields.
    """
    scale = error.detach().mean().clamp_min(1e-6)
    return torch.exp(-error / scale).clamp(0.0, 1.0)


def build_reliability_labels(
    p_img: torch.Tensor,
    y_img: torch.Tensor,
    p_num: torch.Tensor,
    y_num: torch.Tensor,
    tau_img: float = 0.05,
    tau_num: float = 1.0,
    label_mode: str = "mixed",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build reliability supervision labels from tendency prediction errors.

    label_mode:
        absolute:
            exp(-error / tau). This is the original formulation.
        relative:
            exp(-error / batch_mean_error). This is less sensitive to data scale.
        mixed:
            Average of absolute and relative labels. Recommended for multiple
            datasets because it keeps the original meaning while improving
            robustness to different normalization types.
    """
    label_mode = label_mode.lower()

    image_error = ((p_img - y_img) ** 2).mean(dim=(2, 3, 4))
    image_error = image_error.unsqueeze(-1)

    numerical_error = torch.abs(p_num - y_num).mean(dim=-1)
    numerical_error = numerical_error.unsqueeze(-1)

    image_error = torch.nan_to_num(
        image_error,
        nan=1.0,
        posinf=1e6,
        neginf=1e6,
    )

    numerical_error = torch.nan_to_num(
        numerical_error,
        nan=1.0,
        posinf=1e6,
        neginf=1e6,
    )

    abs_img = _absolute_reliability_label(image_error, tau_img)
    abs_num = _absolute_reliability_label(numerical_error, tau_num)

    rel_img = _relative_reliability_label(image_error)
    rel_num = _relative_reliability_label(numerical_error)

    if label_mode == "absolute":
        r_img_label = abs_img
        r_num_label = abs_num

    elif label_mode == "relative":
        r_img_label = rel_img
        r_num_label = rel_num

    elif label_mode == "mixed":
        r_img_label = 0.5 * abs_img + 0.5 * rel_img
        r_num_label = 0.5 * abs_num + 0.5 * rel_num

    else:
        raise ValueError(
            "Unknown reliability label mode: " f"{label_mode}. Use absolute, relative, or mixed."
        )

    return r_img_label.detach(), r_num_label.detach()


def choose_numerical_task_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mae",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Choose numerical task loss while always logging MSE and MAE."""
    loss_type = loss_type.lower()

    if loss_type == "mae":
        return mae_value(pred, target)

    if loss_type == "mse":
        return mse_value(pred, target)

    if loss_type in ["huber", "smooth_l1", "smoothl1"]:
        return huber_value(
            pred=pred,
            target=target,
            beta=huber_delta,
        )

    raise ValueError(f"Unknown numerical loss type: {loss_type}")


def gemininet_loss(
    outputs: dict,
    y_img: torch.Tensor,
    y_num: torch.Tensor,
    lambda_img: float = 1.0,
    lambda_num: float = 1.0,
    lambda_rel: float = 0.1,
    tau_img: float = 0.05,
    tau_num: float = 1.0,
    lambda_img_mae: float = 0.0,
    num_loss_type: str = "mae",
    num_huber_delta: float = 1.0,
    reliability_label_mode: str = "mixed",
) -> tuple[torch.Tensor, dict]:
    """Compute GeminiNet loss and metrics.

    Beta3 keeps the old metrics but allows a more robust numerical task loss
    and an optional small image-MAE auxiliary term.
    """
    image_mse = mse_value(outputs["y_img"], y_img)
    image_mae = mae_value(outputs["y_img"], y_img)
    image_rmse = torch.sqrt(image_mse.clamp_min(0.0))

    num_mse = mse_value(outputs["y_num"], y_num)
    num_mae = mae_value(outputs["y_num"], y_num)
    num_rmse = torch.sqrt(num_mse.clamp_min(0.0))

    num_task_loss = choose_numerical_task_loss(
        pred=outputs["y_num"],
        target=y_num,
        loss_type=num_loss_type,
        huber_delta=num_huber_delta,
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
            label_mode=reliability_label_mode,
        )

        rel_img_loss = F.mse_loss(outputs["r_img"], r_img_label)
        rel_num_loss = F.mse_loss(outputs["r_num"], r_num_label)
        rel_loss = rel_img_loss + rel_num_loss

        r_img_label_mean = r_img_label.mean()
        r_num_label_mean = r_num_label.mean()

    else:
        rel_img_loss = image_mse.new_tensor(0.0)
        rel_num_loss = image_mse.new_tensor(0.0)
        rel_loss = image_mse.new_tensor(0.0)

        r_img_label_mean = image_mse.new_tensor(1.0)
        r_num_label_mean = image_mse.new_tensor(1.0)

    total_loss = (
        lambda_img * image_mse
        + lambda_img_mae * image_mae
        + lambda_num * num_task_loss
        + lambda_rel * rel_loss
    )

    metrics = {
        "total_loss": float(total_loss.detach().cpu()),
        "image_mse": float(image_mse.detach().cpu()),
        "image_mae": float(image_mae.detach().cpu()),
        "num_mse": float(num_mse.detach().cpu()),
        "num_mae": float(num_mae.detach().cpu()),
        "img_mse_norm": float(image_mse.detach().cpu()),
        "img_mae_norm": float(image_mae.detach().cpu()),
        "img_rmse_norm": float(image_rmse.detach().cpu()),
        "num_mse_norm": float(num_mse.detach().cpu()),
        "num_mae_norm": float(num_mae.detach().cpu()),
        "num_rmse_norm": float(num_rmse.detach().cpu()),
        "num_task_loss": float(num_task_loss.detach().cpu()),
        "rel_loss": float(rel_loss.detach().cpu()),
        "rel_img_loss": float(rel_img_loss.detach().cpu()),
        "rel_num_loss": float(rel_num_loss.detach().cpu()),
        "r_img_mean": float(outputs["r_img"].detach().mean().cpu()),
        "r_num_mean": float(outputs["r_num"].detach().mean().cpu()),
        "r_img_label_mean": float(r_img_label_mean.detach().cpu()),
        "r_num_label_mean": float(r_num_label_mean.detach().cpu()),
        "use_reliability_estimator": float(use_reliability_estimator),
    }

    return total_loss, metrics