from __future__ import annotations

import torch
import torch.nn as nn

from .unimodal import (
    SinusoidalPositionalEncoding,
    apply_output_activation,
    clamp_by_activation,
    nchw_to_nhwc_video,
)


class FutureContextExtractor(nn.Module):
    """Extract future-step decoding context.

    Beta2 obtains future context only from cross-attention output. This version
    keeps the tendency-conditioned future query through a residual path and adds
    a small feed-forward block, making it closer to a standard Transformer
    decoder layer:
        query -> cross-attn(history) -> FFN
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.pred_len = pred_len

        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=pred_len + 16,
            dropout=dropout,
        )

        self.query_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.cross_time_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        z_hist: torch.Tensor,
        qp: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        reliable_qp = qp * reliability

        future_position = self.position_encoding(torch.zeros_like(reliable_qp))

        future_query = self.query_mlp(torch.cat([reliable_qp, future_position], dim=-1))

        attention_context, _ = self.cross_time_attention(
            query=future_query,
            key=z_hist,
            value=z_hist,
            need_weights=False,
        )

        context = self.norm1(future_query + self.dropout(attention_context))

        context = self.norm2(context + self.ffn(context))

        return context


class ImageForecastHead(nn.Module):
    """Image forecasting head with tendency-assisted refinement.

    The final prediction contains two parts:
        1. original convex tendency fusion from beta2;
        2. a small learnable residual correction initialized near zero.

    The residual correction allows the model to improve over the frozen image
    prior instead of being strictly limited to a simple interpolation between
    base prediction and tendency prediction.
    """

    def __init__(
        self,
        d_model: int,
        img_channels: int,
        img_size: int = 128,
        base_channels: int = 64,
        dropout: float = 0.1,
        img_output_activation: str = "sigmoid",
        output_scale: float = 5.0,
    ) -> None:
        super().__init__()

        self.img_channels = img_channels
        self.img_size = img_size
        self.base_channels = base_channels
        self.img_output_activation = img_output_activation
        self.output_scale = output_scale

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, base_channels * 8 * 4 * 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 8,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                base_channels,
                img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

        self.tendency_gate = nn.Sequential(
            nn.Conv2d(
                4 * img_channels,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                base_channels,
                img_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.residual_refiner = nn.Sequential(
            nn.Conv2d(
                4 * img_channels,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                base_channels,
                img_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Make the residual refinement branch start from almost no correction.
        last_conv = self.residual_refiner[-2]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.zeros_(last_conv.weight)
            nn.init.zeros_(last_conv.bias)

    def forward(
        self,
        future_context: torch.Tensor,
        p_img: torch.Tensor,
        r_img: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, pred_len, _ = future_context.shape

        x = self.fc(future_context)

        x = x.reshape(
            batch_size * pred_len,
            self.base_channels * 8,
            4,
            4,
        )

        base = apply_output_activation(
            self.decoder(x),
            activation=self.img_output_activation,
            scale=self.output_scale,
        )

        if base.shape[-2:] != (self.img_size, self.img_size):
            base = torch.nn.functional.interpolate(
                base,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        base = base.reshape(
            batch_size,
            pred_len,
            self.img_channels,
            self.img_size,
            self.img_size,
        )

        base = nchw_to_nhwc_video(base)

        reliability_map = r_img.view(batch_size, pred_len, 1, 1, 1)
        reliability_map = reliability_map.expand_as(base)

        gate_input = torch.cat(
            [
                base,
                p_img,
                torch.abs(base - p_img),
                reliability_map,
            ],
            dim=-1,
        )

        gate_input = gate_input.permute(0, 1, 4, 2, 3).contiguous()

        gate_input = gate_input.reshape(
            batch_size * pred_len,
            4 * self.img_channels,
            self.img_size,
            self.img_size,
        )

        tendency_gate = self.tendency_gate(gate_input)
        residual = self.residual_refiner(gate_input) * self.residual_scale

        tendency_gate = tendency_gate.reshape(
            batch_size,
            pred_len,
            self.img_channels,
            self.img_size,
            self.img_size,
        )

        residual = residual.reshape(
            batch_size,
            pred_len,
            self.img_channels,
            self.img_size,
            self.img_size,
        )

        tendency_gate = nchw_to_nhwc_video(tendency_gate)
        residual = nchw_to_nhwc_video(residual)

        prediction = base + tendency_gate * (p_img - base) + residual
        prediction = clamp_by_activation(
            prediction,
            activation=self.img_output_activation,
            scale=self.output_scale,
        )

        return prediction


class NumericalForecastHead(nn.Module):
    """Numerical forecasting head with tendency-assisted residual refinement."""

    def __init__(
        self,
        d_model: int,
        num_vars: int,
        dropout: float = 0.1,
        num_output_activation: str = "linear",
        output_scale: float = 5.0,
    ) -> None:
        super().__init__()

        self.num_output_activation = num_output_activation
        self.output_scale = output_scale

        self.base_predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_vars),
        )

        self.tendency_gate = nn.Sequential(
            nn.Linear(4 * num_vars, 2 * num_vars),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * num_vars, num_vars),
            nn.Sigmoid(),
        )

        self.residual_predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_vars),
            nn.Tanh(),
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        last_linear = self.residual_predictor[-2]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        future_context: torch.Tensor,
        p_num: torch.Tensor,
        r_num: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_predictor(future_context)

        reliability = r_num.expand_as(base)

        gate_input = torch.cat(
            [
                base,
                p_num,
                torch.abs(base - p_num),
                reliability,
            ],
            dim=-1,
        )

        tendency_gate = self.tendency_gate(gate_input)
        residual = self.residual_predictor(future_context) * self.residual_scale

        prediction = base + tendency_gate * (p_num - base) + residual
        prediction = clamp_by_activation(
            prediction,
            activation=self.num_output_activation,
            scale=self.output_scale,
        )

        return prediction