from __future__ import annotations

import math

import torch
import torch.nn as nn


def nhwc_to_nchw_video(x: torch.Tensor) -> torch.Tensor:
    """Convert video tensor from [B, T, H, W, C] to [B, T, C, H, W]."""
    return x.permute(0, 1, 4, 2, 3).contiguous()


def nchw_to_nhwc_video(x: torch.Tensor) -> torch.Tensor:
    """Convert video tensor from [B, T, C, H, W] to [B, T, H, W, C]."""
    return x.permute(0, 1, 3, 4, 2).contiguous()


def apply_output_activation(
    x: torch.Tensor,
    activation: str = "linear",
    scale: float = 5.0,
) -> torch.Tensor:
    """Apply a configurable output activation.

    Use sigmoid for [0, 1] pixel targets, tanh for [-scale, scale] bounded
    targets, and linear for z-score/physical-field targets.
    """
    activation = activation.lower()

    if activation == "linear":
        return x

    if activation == "sigmoid":
        return torch.sigmoid(x)

    if activation == "tanh":
        return torch.tanh(x) * scale

    raise ValueError(f"Unknown output activation: {activation}")


def clamp_by_activation(
    x: torch.Tensor,
    activation: str = "linear",
    scale: float = 5.0,
) -> torch.Tensor:
    """Clamp final predictions only when the target range is bounded."""
    activation = activation.lower()

    if activation == "sigmoid":
        return x.clamp(0.0, 1.0)

    if activation == "tanh":
        return x.clamp(-scale, scale)

    if activation == "linear":
        return x

    raise ValueError(f"Unknown output activation: {activation}")


class DoubleConv(nn.Module):
    """Two convolution blocks used by U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        layers.extend(
            [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageUNetPrior(nn.Module):
    """Image evolutionary tendency generator.

    Input:
        x_img: [B, T_in, H, W, C]

    Output:
        p_img: [B, T_out, H, W, C]

    The U-Net treats the input sequence as stacked channels.
    """

    def __init__(
        self,
        input_len: int = 10,
        pred_len: int = 10,
        img_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.0,
        img_output_activation: str = "sigmoid",
        output_scale: float = 5.0,
    ) -> None:
        super().__init__()

        self.input_len = input_len
        self.pred_len = pred_len
        self.img_channels = img_channels
        self.img_output_activation = img_output_activation
        self.output_scale = output_scale

        input_channels = input_len * img_channels
        output_channels = pred_len * img_channels

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.inc = DoubleConv(input_channels, c1, dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4, dropout))

        self.up1 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(c3 + c3, c3, dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(c2 + c2, c2, dropout)

        self.up3 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(c1 + c1, c1, dropout)

        self.output_conv = nn.Conv2d(c1, output_channels, kernel_size=1)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        batch_size, input_len, height, width, channels = x_img.shape

        if input_len != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got {input_len}.")

        if channels != self.img_channels:
            raise ValueError(f"Expected img_channels={self.img_channels}, got {channels}.")

        # [B,T,H,W,C] -> [B,T,C,H,W] -> [B,T*C,H,W]
        x = nhwc_to_nchw_video(x_img)
        x = x.reshape(
            batch_size,
            input_len * channels,
            height,
            width,
        )

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)

        y = apply_output_activation(
            self.output_conv(x),
            activation=self.img_output_activation,
            scale=self.output_scale,
        )

        y = y.reshape(
            batch_size,
            self.pred_len,
            self.img_channels,
            height,
            width,
        )

        return nchw_to_nhwc_video(y)

    @torch.no_grad()
    def predict_tendency(self, x_img: torch.Tensor) -> torch.Tensor:
        """Generate frozen image tendency during GeminiNet training."""
        self.eval()
        return self.forward(x_img)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


class NumericalTransformerPrior(nn.Module):
    """Numerical evolutionary tendency generator.

    Input:
        x_num: [B, T_in, D], normalized numerical sequence

    Output:
        p_num: [B, T_out, D], normalized numerical tendency
    """

    def __init__(
        self,
        input_len: int = 10,
        pred_len: int = 10,
        num_vars: int = 11,
        d_model: int = 128,
        nhead: int = 4,
        encoder_layers: int = 3,
        decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_output_activation: str = "linear",
        output_scale: float = 5.0,
    ) -> None:
        super().__init__()

        self.input_len = input_len
        self.pred_len = pred_len
        self.num_vars = num_vars
        self.num_output_activation = num_output_activation
        self.output_scale = output_scale

        self.input_projection = nn.Linear(num_vars, d_model)

        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max(input_len, pred_len) + 16,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Learnable future-step queries.
        # Small random initialization helps stabilize early training.
        self.future_queries = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_vars),
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        batch_size, input_len, num_vars = x_num.shape

        if input_len != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got {input_len}.")

        if num_vars != self.num_vars:
            raise ValueError(f"Expected num_vars={self.num_vars}, got {num_vars}.")

        src = self.input_projection(x_num)
        src = self.position_encoding(src)

        memory = self.encoder(src)

        future_query = self.future_queries.unsqueeze(0)
        future_query = future_query.expand(batch_size, -1, -1)
        future_query = self.position_encoding(future_query)

        decoded = self.decoder(
            tgt=future_query,
            memory=memory,
        )

        return apply_output_activation(
            self.output_head(decoded),
            activation=self.num_output_activation,
            scale=self.output_scale,
        )

    @torch.no_grad()
    def predict_tendency(self, x_num: torch.Tensor) -> torch.Tensor:
        """Generate frozen numerical tendency during GeminiNet training."""
        self.eval()
        return self.forward(x_num)