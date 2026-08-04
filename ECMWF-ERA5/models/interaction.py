from __future__ import annotations

import torch
import torch.nn as nn


class ReliabilityAwareTendencyConditioner(nn.Module):
    """Inject reliable tendency features into historical representation.

    This is implemented as a small Transformer-style block:
        historical features query reliability-weighted tendency features,
        then a gated residual attention update and an FFN update are applied.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.tendency_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        hist_features: torch.Tensor,
        tendency_features: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        reliable_tendency = tendency_features * reliability

        tendency_context, _ = self.tendency_attention(
            query=hist_features,
            key=reliable_tendency,
            value=reliable_tendency,
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

        eta = self.gate(gate_input)

        conditioned = self.norm1(hist_features + self.dropout(eta * tendency_context))

        conditioned = self.norm2(conditioned + self.ffn(conditioned))

        return conditioned


class ReliabilityAwareMultimodalInteraction(nn.Module):
    """Reliability-aware tendency-conditioned multimodal interaction.

    Compared with beta2, this block keeps the tendency-conditioned features as
    the residual base, and uses gated bidirectional cross-modal attention in a
    shared space. This makes the interaction less destructive and more adaptive.
    """

    def __init__(
        self,
        d_img: int,
        d_num: int,
        d_shared: int,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.img_conditioner = ReliabilityAwareTendencyConditioner(
            d_model=d_img,
            nhead=nhead,
            dropout=dropout,
        )

        self.num_conditioner = ReliabilityAwareTendencyConditioner(
            d_model=d_num,
            nhead=nhead,
            dropout=dropout,
        )

        self.img_to_shared = nn.Linear(d_img, d_shared)
        self.num_to_shared = nn.Linear(d_num, d_shared)

        self.img_cross_modal_attention = nn.MultiheadAttention(
            embed_dim=d_shared,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.num_cross_modal_attention = nn.MultiheadAttention(
            embed_dim=d_shared,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.img_cross_gate = nn.Sequential(
            nn.Linear(3 * d_shared, 2 * d_shared),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_shared, d_shared),
            nn.Sigmoid(),
        )

        self.num_cross_gate = nn.Sequential(
            nn.Linear(3 * d_shared, 2 * d_shared),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_shared, d_shared),
            nn.Sigmoid(),
        )

        self.shared_dropout = nn.Dropout(dropout)
        self.img_shared_norm1 = nn.LayerNorm(d_shared)
        self.num_shared_norm1 = nn.LayerNorm(d_shared)

        self.img_shared_ffn = nn.Sequential(
            nn.LayerNorm(d_shared),
            nn.Linear(d_shared, 2 * d_shared),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_shared, d_shared),
            nn.Dropout(dropout),
        )

        self.num_shared_ffn = nn.Sequential(
            nn.LayerNorm(d_shared),
            nn.Linear(d_shared, 2 * d_shared),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_shared, d_shared),
            nn.Dropout(dropout),
        )

        self.img_shared_norm2 = nn.LayerNorm(d_shared)
        self.num_shared_norm2 = nn.LayerNorm(d_shared)

        self.shared_to_img = nn.Linear(d_shared, d_img)
        self.shared_to_num = nn.Linear(d_shared, d_num)

        self.output_dropout = nn.Dropout(dropout)
        self.img_output_norm = nn.LayerNorm(d_img)
        self.num_output_norm = nn.LayerNorm(d_num)

    def forward(
        self,
        e_img: torch.Tensor,
        e_num: torch.Tensor,
        qp_img: torch.Tensor,
        qp_num: torch.Tensor,
        r_img: torch.Tensor,
        r_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_conditioned = self.img_conditioner(e_img, qp_img, r_img)
        num_conditioned = self.num_conditioner(e_num, qp_num, r_num)

        img_shared = self.img_to_shared(img_conditioned)
        num_shared = self.num_to_shared(num_conditioned)

        img_interaction, _ = self.img_cross_modal_attention(
            query=img_shared,
            key=num_shared,
            value=num_shared,
            need_weights=False,
        )

        num_interaction, _ = self.num_cross_modal_attention(
            query=num_shared,
            key=img_shared,
            value=img_shared,
            need_weights=False,
        )

        img_gate = self.img_cross_gate(
            torch.cat(
                [
                    img_shared,
                    img_interaction,
                    torch.abs(img_shared - img_interaction),
                ],
                dim=-1,
            )
        )

        num_gate = self.num_cross_gate(
            torch.cat(
                [
                    num_shared,
                    num_interaction,
                    torch.abs(num_shared - num_interaction),
                ],
                dim=-1,
            )
        )

        img_shared = self.img_shared_norm1(
            img_shared + self.shared_dropout(img_gate * img_interaction)
        )

        num_shared = self.num_shared_norm1(
            num_shared + self.shared_dropout(num_gate * num_interaction)
        )

        img_shared = self.img_shared_norm2(img_shared + self.img_shared_ffn(img_shared))

        num_shared = self.num_shared_norm2(num_shared + self.num_shared_ffn(num_shared))

        z_img = self.img_output_norm(
            img_conditioned + self.output_dropout(self.shared_to_img(img_shared))
        )

        z_num = self.num_output_norm(
            num_conditioned + self.output_dropout(self.shared_to_num(num_shared))
        )

        return z_img, z_num