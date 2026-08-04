from __future__ import annotations

import torch
import torch.nn as nn

from .unimodal import (
    SinusoidalPositionalEncoding,
    nhwc_to_nchw_video,
)


class FiLMConditioning(nn.Module):
    """Stable residual tendency conditioning.

    Beta2 uses an unconstrained FiLM-style modulation:
        E = H + alpha * (gamma * H + beta)

    This version keeps the same idea, but makes the conditioning safer and
    easier to optimize:
        1. gamma and beta are bounded with tanh;
        2. alpha is initialized small;
        3. the modulation branch is zero-initialized, so the module starts
           close to an identity mapping.

    This is important because the tendency is only a weak prior. It should help
    the encoder, not overwrite historical evidence at the beginning of training.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        max_modulation: float = 0.25,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model * 2

        self.max_modulation = max_modulation

        self.summary_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
        )

        self.gamma_beta = nn.Linear(hidden_dim, 2 * d_model)

        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

        self.output_norm = nn.LayerNorm(d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Start from identity conditioning.
        nn.init.zeros_(self.gamma_beta.weight)
        nn.init.zeros_(self.gamma_beta.bias)

        # alpha starts small: sigmoid(-2) ~= 0.12
        alpha_linear = self.alpha_net[0]
        nn.init.xavier_uniform_(alpha_linear.weight)
        nn.init.constant_(alpha_linear.bias, -2.0)

    def forward(
        self,
        hist_features: torch.Tensor,
        tendency_features: torch.Tensor,
    ) -> torch.Tensor:
        summary = tendency_features.mean(dim=1)
        summary = self.summary_net(summary)

        gamma_beta = self.gamma_beta(summary)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = self.max_modulation * torch.tanh(gamma)
        beta = self.max_modulation * torch.tanh(beta)
        alpha = self.alpha_net(summary)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        alpha = alpha.unsqueeze(1)

        conditioned = hist_features + alpha * (gamma * hist_features + beta)

        conditioned = self.output_norm(conditioned)

        return conditioned


class NumericalTendencyConditionedEncoder(nn.Module):
    """Numerical branch of Tendency-Conditioned Representation Learning."""

    def __init__(
        self,
        num_vars: int,
        d_model: int,
        input_len: int,
        pred_len: int,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hist_projection = nn.Linear(num_vars, d_model)
        self.tendency_projection = nn.Linear(num_vars, d_model)

        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max(input_len, pred_len) + 16,
            dropout=dropout,
        )

        hist_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        tendency_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.hist_encoder = nn.TransformerEncoder(
            encoder_layer=hist_layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_model),
        )

        self.tendency_encoder = nn.TransformerEncoder(
            encoder_layer=tendency_layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_model),
        )

        self.conditioning = FiLMConditioning(d_model)

    def forward(
        self,
        x_num: torch.Tensor,
        p_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hist = self.hist_projection(x_num)
        hist = self.position_encoding(hist)
        hist = self.hist_encoder(hist)

        tendency = self.tendency_projection(p_num)
        tendency = self.position_encoding(tendency)
        tendency = self.tendency_encoder(tendency)

        conditioned = self.conditioning(
            hist_features=hist,
            tendency_features=tendency,
        )

        return conditioned, tendency


class FrameCNNTokenEncoder(nn.Module):
    """Encode each frame into one temporal token.

    Input:
        x_img: [B, T, H, W, C]

    Output:
        tokens: [B, T, D]
    """

    def __init__(
        self,
        img_channels: int,
        d_model: int,
        base_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                img_channels,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, height, width, channels = x_img.shape

        x = nhwc_to_nchw_video(x_img)
        x = x.reshape(batch_size * seq_len, channels, height, width)

        token = self.projection(self.cnn(x))
        token = token.reshape(batch_size, seq_len, -1)

        return token


class ImageTendencyConditionedEncoder(nn.Module):
    """Image branch of Tendency-Conditioned Representation Learning."""

    def __init__(
        self,
        img_channels: int,
        d_model: int,
        input_len: int,
        pred_len: int,
        base_channels: int = 32,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hist_frame_encoder = FrameCNNTokenEncoder(
            img_channels=img_channels,
            d_model=d_model,
            base_channels=base_channels,
            dropout=dropout,
        )

        self.tendency_frame_encoder = FrameCNNTokenEncoder(
            img_channels=img_channels,
            d_model=d_model,
            base_channels=base_channels,
            dropout=dropout,
        )

        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max(input_len, pred_len) + 16,
            dropout=dropout,
        )

        hist_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        tendency_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.hist_temporal_encoder = nn.TransformerEncoder(
            encoder_layer=hist_layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_model),
        )

        self.tendency_temporal_encoder = nn.TransformerEncoder(
            encoder_layer=tendency_layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_model),
        )

        self.conditioning = FiLMConditioning(d_model)

    def forward(
        self,
        x_img: torch.Tensor,
        p_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hist = self.hist_frame_encoder(x_img)
        hist = self.position_encoding(hist)
        hist = self.hist_temporal_encoder(hist)

        tendency = self.tendency_frame_encoder(p_img)
        tendency = self.position_encoding(tendency)
        tendency = self.tendency_temporal_encoder(tendency)

        conditioned = self.conditioning(
            hist_features=hist,
            tendency_features=tendency,
        )

        return conditioned, tendency