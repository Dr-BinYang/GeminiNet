from __future__ import annotations

import torch
import torch.nn as nn


class TendencyReliabilityEstimator(nn.Module):
    """Estimate the reliability of evolutionary tendency at each future step.

    The estimator aligns future tendency features with historical dynamics.
    If the tendency representation is compatible with historical context,
    the predicted reliability should be high.
    """

    def __init__(
        self,
        d_img: int,
        d_num: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.img_cross_time_attention = nn.MultiheadAttention(
            embed_dim=d_img,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.num_cross_time_attention = nn.MultiheadAttention(
            embed_dim=d_num,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.img_head = nn.Sequential(
            nn.Linear(4 * d_img, 2 * d_img),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_img, 1),
            nn.Sigmoid(),
        )

        self.num_head = nn.Sequential(
            nn.Linear(4 * d_num, 2 * d_num),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_num, 1),
            nn.Sigmoid(),
        )

    def _estimate_one_modality(
        self,
        tendency_features: torch.Tensor,
        hist_features: torch.Tensor,
        attention: nn.MultiheadAttention,
        head: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Cross-time alignment:
        # future tendency features query historical dynamic features.
        context, _ = attention(
            query=tendency_features,
            key=hist_features,
            value=hist_features,
            need_weights=False,
        )

        consistency = torch.cat(
            [
                tendency_features,
                context,
                torch.abs(tendency_features - context),
                tendency_features * context,
            ],
            dim=-1,
        )

        reliability = head(consistency)

        return reliability, tendency_features

    def forward(
        self,
        e_img: torch.Tensor,
        e_num: torch.Tensor,
        q_img: torch.Tensor,
        q_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_img, qp_img = self._estimate_one_modality(
            tendency_features=q_img,
            hist_features=e_img,
            attention=self.img_cross_time_attention,
            head=self.img_head,
        )

        r_num, qp_num = self._estimate_one_modality(
            tendency_features=q_num,
            hist_features=e_num,
            attention=self.num_cross_time_attention,
            head=self.num_head,
        )

        return r_img, r_num, qp_img, qp_num