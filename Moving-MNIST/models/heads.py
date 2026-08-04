from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unimodal import (
    SinusoidalPositionalEncoding,
    apply_output_activation,
    nchw_to_nhwc_video,
    validate_output_activation,
)


class FutureContextExtractor(nn.Module):
    """Extract future-step decoding context from global multimodal memory."""

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        nhead: int = 4,
        dropout: float = 0.1,
        memory_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.pred_len = pred_len
        self.memory_dim = memory_dim if memory_dim is not None else d_model

        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=pred_len + 16,
            dropout=dropout,
        )

        self.query_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.cross_memory_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            kdim=self.memory_dim,
            vdim=self.memory_dim,
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        z_hist: torch.Tensor,
        qp: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        # reliability controls two complementary signals:
        #   reliable tendency token: r * qp
        #   uncertainty token: (1-r) * qp
        reliable_qp = qp * reliability
        uncertain_qp = qp * (1.0 - reliability)

        future_position = self.position_encoding(torch.zeros_like(qp))

        future_query = self.query_mlp(
            torch.cat(
                [
                    reliable_qp,
                    uncertain_qp,
                    future_position,
                ],
                dim=-1,
            )
        )

        future_context, _ = self.cross_memory_attention(
            query=future_query,
            key=z_hist,
            value=z_hist,
            need_weights=False,
        )

        future_context = self.norm1(future_query + future_context)
        future_context = self.norm2(future_context + self.ffn(future_context))

        return future_context


class ImageForecastHead(nn.Module):
    """Image forecasting head with reliability-aware prior residual refinement."""

    def __init__(
        self,
        d_model: int,
        img_channels: int,
        img_size: int = 64,
        base_channels: int = 64,
        img_output_activation: str = "sigmoid",
        output_scale: float = 5.0,
    ) -> None:
        super().__init__()

        self.img_channels = img_channels
        self.img_size = img_size
        self.base_channels = base_channels
        self.img_output_activation = validate_output_activation(
            img_output_activation,
            allowed={"sigmoid", "tanh", "linear"},
        )
        self.output_scale = output_scale

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, base_channels * 8 * 4 * 4),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, img_channels, 4, stride=2, padding=1),
        )

        self.tendency_gate = nn.Sequential(
            nn.Conv2d(4 * img_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, img_channels, 1),
            nn.Sigmoid(),
        )

        self.residual_refiner = nn.Sequential(
            nn.Conv2d(4 * img_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, img_channels, 1),
        )

        self._init_residual_refiner()

    def _init_residual_refiner(self) -> None:
        final_conv = self.residual_refiner[-1]
        if isinstance(final_conv, nn.Conv2d):
            nn.init.zeros_(final_conv.weight)
            nn.init.zeros_(final_conv.bias)

    def forward(
        self,
        future_context: torch.Tensor,
        p_img: torch.Tensor,
        r_img: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, pred_len, _ = future_context.shape

        x = self.fc(future_context)
        x = x.reshape(batch_size * pred_len, self.base_channels * 8, 4, 4)

        raw_base = self.decoder(x)

        if raw_base.shape[-2:] != (self.img_size, self.img_size):
            raw_base = F.interpolate(
                raw_base,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        base = apply_output_activation(
            raw=raw_base,
            activation=self.img_output_activation,
            output_scale=self.output_scale,
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
        raw_residual = self.residual_refiner(gate_input)

        if self.img_output_activation == "linear":
            residual = raw_residual
        elif self.img_output_activation == "tanh":
            residual = torch.tanh(raw_residual) * self.output_scale
        else:
            residual = torch.tanh(raw_residual)

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

        blended = base + tendency_gate * (p_img - base)

        # The residual is deliberately small at the beginning and bounded later.
        # Low reliability allows larger correction; high reliability preserves prior.
        correction_scale = 0.1 + 0.2 * (1.0 - reliability_map)
        prediction = blended + correction_scale * residual

        if self.img_output_activation == "sigmoid":
            prediction = prediction.clamp(0.0, 1.0)
        elif self.img_output_activation == "tanh":
            prediction = prediction.clamp(
                -self.output_scale,
                self.output_scale,
            )

        return prediction


class NumericalForecastHead(nn.Module):
    """Numerical forecasting head with reliability-aware prior residual refinement."""

    def __init__(
        self,
        d_model: int,
        num_vars: int,
        dropout: float = 0.1,
        num_output_activation: str = "linear",
        output_scale: float = 5.0,
    ) -> None:
        super().__init__()

        self.num_output_activation = validate_output_activation(
            num_output_activation,
            allowed={"linear", "tanh"},
        )
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

        self.residual_refiner = nn.Sequential(
            nn.Linear(4 * num_vars, 4 * num_vars),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * num_vars, num_vars),
        )

        self._init_residual_refiner()

    def _init_residual_refiner(self) -> None:
        final_linear = self.residual_refiner[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        future_context: torch.Tensor,
        p_num: torch.Tensor,
        r_num: torch.Tensor,
    ) -> torch.Tensor:
        base = apply_output_activation(
            raw=self.base_predictor(future_context),
            activation=self.num_output_activation,
            output_scale=self.output_scale,
        )

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
        raw_residual = self.residual_refiner(gate_input)

        if self.num_output_activation == "linear":
            residual = raw_residual
        else:
            residual = torch.tanh(raw_residual) * self.output_scale

        blended = base + tendency_gate * (p_num - base)

        correction_scale = 0.1 + 0.4 * (1.0 - reliability)
        prediction = blended + correction_scale * residual

        if self.num_output_activation == "tanh":
            prediction = prediction.clamp(
                -self.output_scale,
                self.output_scale,
            )

        return prediction