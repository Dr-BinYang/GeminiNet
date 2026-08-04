from __future__ import annotations

import torch
import torch.nn as nn


class ReliabilityAwareMultimodalInteraction(nn.Module):
    """Reliability-aware global multimodal token fusion.

    The previous implementation used pairwise image-to-number and number-to-
    image attention. This global version builds a single token memory composed
    of historical tokens and calibrated tendency tokens from both modalities:

        image history tokens
        numerical history tokens
        image tendency tokens with reliability embedding
        numerical tendency tokens with reliability embedding

    A Transformer encoder then performs global interaction across all tokens.
    """

    def __init__(
        self,
        d_img: int,
        d_num: int,
        d_shared: int,
        nhead: int = 4,
        dropout: float = 0.1,
        layers: int = 2,
    ) -> None:
        super().__init__()

        self.d_shared = d_shared

        self.img_hist_to_shared = nn.Linear(d_img, d_shared)
        self.num_hist_to_shared = nn.Linear(d_num, d_shared)
        self.img_tendency_to_shared = nn.Linear(d_img, d_shared)
        self.num_tendency_to_shared = nn.Linear(d_num, d_shared)
        self.img_specific_to_shared = nn.Linear(d_img, d_shared)
        self.num_specific_to_shared = nn.Linear(d_num, d_shared)

        self.reliability_embedding = nn.Sequential(
            nn.Linear(1, d_shared),
            nn.GELU(),
            nn.Linear(d_shared, d_shared),
        )

        self.uncertainty_embedding = nn.Sequential(
            nn.Linear(1, d_shared),
            nn.GELU(),
            nn.Linear(d_shared, d_shared),
        )

        self.type_embedding = nn.Parameter(torch.zeros(4, d_shared))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_shared,
            nhead=nhead,
            dim_feedforward=d_shared * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.global_fusion = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_shared),
        )

        self.memory_ffn = nn.Sequential(
            nn.LayerNorm(d_shared),
            nn.Linear(d_shared, 2 * d_shared),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_shared, d_shared),
        )

        self.img_output = nn.Sequential(
            nn.Linear(d_shared, d_img),
            nn.LayerNorm(d_img),
        )

        self.num_output = nn.Sequential(
            nn.Linear(d_shared, d_num),
            nn.LayerNorm(d_num),
        )

    def _add_type(
        self,
        tokens: torch.Tensor,
        type_index: int,
    ) -> torch.Tensor:
        return tokens + self.type_embedding[type_index].view(1, 1, -1)

    def project_modality_memory(
        self,
        z_img: torch.Tensor,
        z_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project modality-specific history tokens to the shared memory space.

        Inputs:
            z_img: [B, T_img, d_img]
            z_num: [B, T_num, d_num]

        Outputs:
            img_memory: [B, T_img, d_shared]
            num_memory: [B, T_num, d_shared]
        """
        img_memory = self.img_specific_to_shared(z_img)
        num_memory = self.num_specific_to_shared(z_num)

        return img_memory, num_memory

    def forward(
        self,
        e_img: torch.Tensor,
        e_num: torch.Tensor,
        qp_img: torch.Tensor,
        qp_num: torch.Tensor,
        r_img: torch.Tensor,
        r_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse both modalities and return specific and global memories.

        Returns:
            z_img: [B, T_img, d_img]
            z_num: [B, T_num, d_num]
            memory: [B, T_img + T_num + P_img + P_num, d_shared]
        """
        img_hist = self._add_type(
            self.img_hist_to_shared(e_img),
            type_index=0,
        )

        num_hist = self._add_type(
            self.num_hist_to_shared(e_num),
            type_index=1,
        )

        img_rel = self.reliability_embedding(r_img)
        img_unc = self.uncertainty_embedding(1.0 - r_img)
        img_tendency = self._add_type(
            self.img_tendency_to_shared(qp_img) + img_rel + img_unc,
            type_index=2,
        )

        num_rel = self.reliability_embedding(r_num)
        num_unc = self.uncertainty_embedding(1.0 - r_num)
        num_tendency = self._add_type(
            self.num_tendency_to_shared(qp_num) + num_rel + num_unc,
            type_index=3,
        )

        memory_input = torch.cat(
            [
                img_hist,
                num_hist,
                img_tendency,
                num_tendency,
            ],
            dim=1,
        )

        memory = self.global_fusion(memory_input)
        memory = memory + self.memory_ffn(memory)

        img_len = e_img.size(1)
        num_len = e_num.size(1)

        img_memory = memory[:, :img_len, :]
        num_memory = memory[:, img_len : img_len + num_len, :]

        z_img = e_img + self.img_output(img_memory)
        z_num = e_num + self.num_output(num_memory)

        return z_img, z_num, memory