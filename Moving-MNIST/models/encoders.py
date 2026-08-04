from __future__ import annotations

import torch
import torch.nn as nn

from .unimodal import (
    SinusoidalPositionalEncoding,
    nhwc_to_nchw_video,
)


class CrossTendencyConditioning(nn.Module):
    """Condition historical tokens with full future tendency tokens.

    The previous implementation compressed all future tendency tokens into one
    global summary and then used FiLM modulation. That is stable, but it loses
    step-wise future information. This module lets each historical token attend
    to all tendency tokens, while keeping the update residual and gated.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
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
        hist_features: torch.Tensor,
        tendency_features: torch.Tensor,
    ) -> torch.Tensor:
        tendency_context, _ = self.cross_attention(
            query=hist_features,
            key=tendency_features,
            value=tendency_features,
            need_weights=False,
        )

        gate_input = torch.cat(
            [
                hist_features,
                tendency_context,
                torch.abs(hist_features - tendency_context),
            ],
            dim=-1,
        )

        gate = self.gate(gate_input)
        conditioned = self.norm1(hist_features + gate * tendency_context)
        conditioned = self.norm2(conditioned + self.ffn(conditioned))

        return conditioned


class NumericalTendencyConditionedEncoder(nn.Module):
    """Numerical encoder with step-wise tendency-aware conditioning."""

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

        self.conditioning = CrossTendencyConditioning(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

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
    """Encode each image frame into one spatially aware temporal token.

    Input:
        x_img: [B, T, H, W, C]

    Output:
        tokens: [B, T, D]

    A small spatial grid is flattened instead of globally averaging the frame.
    This preserves coarse object/field location while retaining the temporal
    token interface expected by the rest of GeminiNet.
    """

    def __init__(
        self,
        img_channels: int,
        d_model: int,
        base_channels: int = 32,
        spatial_grid: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_grid = spatial_grid

        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((spatial_grid, spatial_grid)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(
                base_channels * 4 * spatial_grid * spatial_grid,
                d_model,
            ),
        )

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, height, width, channels = x_img.shape

        x = nhwc_to_nchw_video(x_img)
        x = x.reshape(batch_size * seq_len, channels, height, width)

        token = self.projection(self.cnn(x))
        token = token.reshape(batch_size, seq_len, -1)

        return token


class ImageTendencyConditionedEncoder(nn.Module):
    """Image encoder with step-wise tendency-aware conditioning."""

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

        self.conditioning = CrossTendencyConditioning(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

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