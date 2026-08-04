from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import (
    ImageTendencyConditionedEncoder,
    NumericalTendencyConditionedEncoder,
)
from .heads import (
    FutureContextExtractor,
    ImageForecastHead,
    NumericalForecastHead,
)
from .interaction import ReliabilityAwareMultimodalInteraction
from .reliability import TendencyReliabilityEstimator
from .unimodal import ImageUNetPrior, NumericalTransformerPrior


class EvolutionaryTendencyGeneration(nn.Module):
    """Frozen unimodal tendency generators."""

    def __init__(
        self,
        image_prior: ImageUNetPrior,
        num_prior: NumericalTransformerPrior,
    ) -> None:
        super().__init__()

        self.image_prior = image_prior
        self.num_prior = num_prior

        self.freeze()

    def freeze(self) -> None:
        for model in [self.image_prior, self.num_prior]:
            model.eval()

            for parameter in model.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        x_img: torch.Tensor,
        x_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.image_prior.eval()
        self.num_prior.eval()

        p_img = self.image_prior.predict_tendency(x_img)
        p_num = self.num_prior.predict_tendency(x_num)

        return p_img, p_num


class GeminiNet(nn.Module):
    """GeminiNet beta5: global fusion with calibrated reliability.

    Main differences from beta2:
        1. Step-wise tendency conditioning instead of global-summary FiLM.
        2. Reliability estimator calibrates tendency features instead of only
           predicting scalar scores.
        3. Global multimodal token memory fuses history and tendency tokens from
           both modalities.
        4. Forecast heads use reliability-aware prior residual refinement.
    """

    def __init__(
        self,
        image_prior: ImageUNetPrior,
        num_prior: NumericalTransformerPrior,
        input_len_img: int = 10,
        input_len_num: int = 10,
        pred_len_img: int = 10,
        pred_len_num: int = 10,
        img_size: int = 64,
        img_channels: int = 1,
        num_vars: int = 11,
        d_img: int = 128,
        d_num: int = 128,
        d_shared: int = 128,
        nhead: int = 4,
        dropout: float = 0.1,
        use_reliability_estimator: bool = True,
        img_output_activation: str = "sigmoid",
        num_output_activation: str = "linear",
        output_scale: float = 5.0,
    ) -> None:
        super().__init__()

        self.use_reliability_estimator = use_reliability_estimator

        self.tendency_generator = EvolutionaryTendencyGeneration(
            image_prior=image_prior,
            num_prior=num_prior,
        )

        self.img_encoder = ImageTendencyConditionedEncoder(
            img_channels=img_channels,
            d_model=d_img,
            input_len=input_len_img,
            pred_len=pred_len_img,
            nhead=nhead,
            dropout=dropout,
        )

        self.num_encoder = NumericalTendencyConditionedEncoder(
            num_vars=num_vars,
            d_model=d_num,
            input_len=input_len_num,
            pred_len=pred_len_num,
            nhead=nhead,
            dropout=dropout,
        )

        self.reliability_estimator = TendencyReliabilityEstimator(
            d_img=d_img,
            d_num=d_num,
            nhead=nhead,
            dropout=dropout,
        )

        self.multimodal_interaction = ReliabilityAwareMultimodalInteraction(
            d_img=d_img,
            d_num=d_num,
            d_shared=d_shared,
            nhead=nhead,
            dropout=dropout,
        )

        self.img_context_extractor = FutureContextExtractor(
            d_model=d_img,
            memory_dim=d_shared,
            pred_len=pred_len_img,
            nhead=nhead,
            dropout=dropout,
        )

        self.num_context_extractor = FutureContextExtractor(
            d_model=d_num,
            memory_dim=d_shared,
            pred_len=pred_len_num,
            nhead=nhead,
            dropout=dropout,
        )

        self.img_head = ImageForecastHead(
            d_model=d_img,
            img_channels=img_channels,
            img_size=img_size,
            img_output_activation=img_output_activation,
            output_scale=output_scale,
        )

        self.num_head = NumericalForecastHead(
            d_model=d_num,
            num_vars=num_vars,
            dropout=dropout,
            num_output_activation=num_output_activation,
            output_scale=output_scale,
        )

    @staticmethod
    def build_default_reliability(
        q_img: torch.Tensor,
        q_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r_img = torch.ones(
            q_img.size(0),
            q_img.size(1),
            1,
            device=q_img.device,
            dtype=q_img.dtype,
        )

        r_num = torch.ones(
            q_num.size(0),
            q_num.size(1),
            1,
            device=q_num.device,
            dtype=q_num.dtype,
        )

        return r_img, r_num

    def forward(
        self,
        x_img: torch.Tensor,
        x_num: torch.Tensor,
    ) -> dict:
        p_img, p_num = self.tendency_generator(x_img, x_num)

        e_img, q_img = self.img_encoder(x_img, p_img)
        e_num, q_num = self.num_encoder(x_num, p_num)

        if self.use_reliability_estimator:
            r_img, r_num, qp_img, qp_num = self.reliability_estimator(
                e_img=e_img,
                e_num=e_num,
                q_img=q_img,
                q_num=q_num,
            )
        else:
            # Ablation only. Default beta5 uses reliability estimator.
            r_img, r_num = self.build_default_reliability(
                q_img=q_img,
                q_num=q_num,
            )
            qp_img = q_img
            qp_num = q_num

        z_img, z_num, global_memory = self.multimodal_interaction(
            e_img=e_img,
            e_num=e_num,
            qp_img=qp_img,
            qp_num=qp_num,
            r_img=r_img,
            r_num=r_num,
        )

        img_specific_memory, num_specific_memory = (
            self.multimodal_interaction.project_modality_memory(
                z_img=z_img,
                z_num=z_num,
            )
        )

        img_memory = torch.cat(
            [img_specific_memory, global_memory],
            dim=1,
        )
        num_memory = torch.cat(
            [num_specific_memory, global_memory],
            dim=1,
        )

        d_img = self.img_context_extractor(
            z_hist=img_memory,
            qp=qp_img,
            reliability=r_img,
        )

        d_num = self.num_context_extractor(
            z_hist=num_memory,
            qp=qp_num,
            reliability=r_num,
        )

        y_img = self.img_head(
            future_context=d_img,
            p_img=p_img,
            r_img=r_img,
        )

        y_num = self.num_head(
            future_context=d_num,
            p_num=p_num,
            r_num=r_num,
        )

        return {
            "y_img": y_img,
            "y_num": y_num,
            "p_img": p_img,
            "p_num": p_num,
            "r_img": r_img,
            "r_num": r_num,
            "qp_img": qp_img,
            "qp_num": qp_num,
            "z_img": z_img,
            "z_num": z_num,
            "global_memory": global_memory,
            "use_reliability_estimator": self.use_reliability_estimator,
        }