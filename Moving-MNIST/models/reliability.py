from __future__ import annotations

import torch
import torch.nn as nn


class CalibratedReliabilityBlock(nn.Module):
    """Estimate tendency reliability and calibrate tendency features.

    Reliability should not be a fragile stand-alone score. This block first
    aligns future tendency tokens with historical dynamics, then uses the
    alignment residual to produce two outputs:

        1. calibrated tendency token qp
        2. reliability score r in [0, 1]

    A high r means the tendency token is consistent with historical dynamics;
    a low r means the downstream fusion/forecasting modules should allow larger
    correction from multimodal context.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cross_time_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.calibration_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.calibrated_norm = nn.LayerNorm(d_model)

        self.reliability_head = nn.Sequential(
            nn.Linear(5 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_reliability_head()

    def _init_reliability_head(self) -> None:
        """Start with mildly optimistic reliability, not saturated scores."""
        final_linear = self.reliability_head[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, 0.5)

    def forward(
        self,
        tendency_features: torch.Tensor,
        hist_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hist_context, _ = self.cross_time_attention(
            query=tendency_features,
            key=hist_features,
            value=hist_features,
            need_weights=False,
        )

        delta = hist_context - tendency_features

        gate = self.calibration_gate(
            torch.cat(
                [
                    tendency_features,
                    hist_context,
                    torch.abs(delta),
                ],
                dim=-1,
            )
        )

        qp = self.calibrated_norm(tendency_features + gate * delta)

        reliability_input = torch.cat(
            [
                tendency_features,
                hist_context,
                qp,
                torch.abs(tendency_features - hist_context),
                tendency_features * hist_context,
            ],
            dim=-1,
        )

        reliability = torch.sigmoid(self.reliability_head(reliability_input))

        return reliability, qp


class TendencyReliabilityEstimator(nn.Module):
    """Estimate reliability and calibrate image and numerical tendency tokens."""

    def __init__(
        self,
        d_img: int,
        d_num: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.img_block = CalibratedReliabilityBlock(
            d_model=d_img,
            nhead=nhead,
            dropout=dropout,
        )

        self.num_block = CalibratedReliabilityBlock(
            d_model=d_num,
            nhead=nhead,
            dropout=dropout,
        )

    def forward(
        self,
        e_img: torch.Tensor,
        e_num: torch.Tensor,
        q_img: torch.Tensor,
        q_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_img, qp_img = self.img_block(
            tendency_features=q_img,
            hist_features=e_img,
        )

        r_num, qp_num = self.num_block(
            tendency_features=q_num,
            hist_features=e_num,
        )

        return r_img, r_num, qp_img, qp_num