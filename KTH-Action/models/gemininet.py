from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Frozen unimodal tendency generators.

    The image prior model and numerical prior model are trained before
    GeminiNet. During GeminiNet training, they are frozen and only provide
    tendency predictions.
    """

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
        """Freeze the two unimodal tendency models."""
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
        """Generate image and numerical tendency predictions.

        Both prior models expose a shared predict_tendency() interface.
        """
        self.image_prior.eval()
        self.num_prior.eval()

        p_img = self.image_prior.predict_tendency(x_img)
        p_num = self.num_prior.predict_tendency(x_num)

        return p_img, p_num


class GeminiNet(nn.Module):
    """GeminiNet main model with optional reliability estimation.

    Args:
        use_reliability_estimator:
            If True, use TendencyReliabilityEstimator to estimate r_img and r_num.
            If False, skip TendencyReliabilityEstimator and set r_img = 1,
            r_num = 1, which means all tendency predictions are regarded as
            fully reliable.

    Input:
        x_img: [B, T_h_img, H, W, C]
        x_num: [B, T_h_num, D], normalized

    Output dictionary:
        y_img: final image prediction
        y_num: final numerical prediction
        p_img: frozen image tendency
        p_num: frozen numerical tendency
        r_img: image tendency reliability
        r_num: numerical tendency reliability
        use_reliability_estimator: reliability-estimator switch
    """

    def __init__(
        self,
        image_prior: ImageUNetPrior,
        num_prior: NumericalTransformerPrior,
        input_len_img: int = 10,
        input_len_num: int = 10,
        pred_len_img: int = 10,
        pred_len_num: int = 10,
        img_height: int = 240,
        img_width: int = 320,
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
        self.img_output_activation = img_output_activation
        self.num_output_activation = num_output_activation
        self.output_scale = output_scale

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

        # Always keep this module for checkpoint compatibility.
        # In forward(), it is skipped when use_reliability_estimator=False.
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
            pred_len=pred_len_img,
            nhead=nhead,
            dropout=dropout,
        )

        self.num_context_extractor = FutureContextExtractor(
            d_model=d_num,
            pred_len=pred_len_num,
            nhead=nhead,
            dropout=dropout,
        )

        self.img_head = ImageForecastHead(
            d_model=d_img,
            img_channels=img_channels,
            img_height=img_height,
            img_width=img_width,
            dropout=dropout,
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
    def resize_video_nhwc(
        video: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Resize a NHWC video tensor while preserving batch/time/channel axes."""
        if video.shape[2] == height and video.shape[3] == width:
            return video

        batch_size, time_steps, old_height, old_width, channels = video.shape

        video_nchw = video.permute(0, 1, 4, 2, 3).contiguous()
        video_nchw = video_nchw.reshape(
            batch_size * time_steps,
            channels,
            old_height,
            old_width,
        )

        video_nchw = F.interpolate(
            video_nchw,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        video = video_nchw.reshape(
            batch_size,
            time_steps,
            channels,
            height,
            width,
        )

        return video.permute(0, 1, 3, 4, 2).contiguous()

    @staticmethod
    def build_default_reliability(
        q_img: torch.Tensor,
        q_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build all-one reliability scores.

        Args:
            q_img:
                Image tendency feature, shape [B, pred_len_img, d_img].

            q_num:
                Numerical tendency feature, shape [B, pred_len_num, d_num].

        Returns:
            r_img:
                All-one image reliability, shape [B, pred_len_img, 1].

            r_num:
                All-one numerical reliability, shape [B, pred_len_num, 1].
        """
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
        # Module 2: frozen tendency generation.
        p_img, p_num = self.tendency_generator(x_img, x_num)
        p_img = self.resize_video_nhwc(
            video=p_img,
            height=x_img.shape[2],
            width=x_img.shape[3],
        )

        # Module 3: tendency-conditioned representation learning.
        e_img, q_img = self.img_encoder(x_img, p_img)
        e_num, q_num = self.num_encoder(x_num, p_num)

        # Module 4: optional tendency reliability estimation.
        if self.use_reliability_estimator:
            r_img, r_num, qp_img, qp_num = self.reliability_estimator(
                e_img=e_img,
                e_num=e_num,
                q_img=q_img,
                q_num=q_num,
            )

        else:
            # Ablation mode:
            #   skip TendencyReliabilityEstimator;
            #   regard all tendency predictions as fully reliable.
            r_img, r_num = self.build_default_reliability(
                q_img=q_img,
                q_num=q_num,
            )

            qp_img = q_img
            qp_num = q_num

        # Module 5: reliability-aware multimodal interaction.
        # When reliability is all-one, this becomes tendency-conditioned
        # interaction without estimated reliability suppression.
        z_img, z_num = self.multimodal_interaction(
            e_img=e_img,
            e_num=e_num,
            qp_img=qp_img,
            qp_num=qp_num,
            r_img=r_img,
            r_num=r_num,
        )

        # Module 6.1: future-step context extraction.
        d_img = self.img_context_extractor(
            z_hist=z_img,
            qp=qp_img,
            reliability=r_img,
        )

        d_num = self.num_context_extractor(
            z_hist=z_num,
            qp=qp_num,
            reliability=r_num,
        )

        # Module 6.2 and 6.3: multitask forecasting heads.
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
            "use_reliability_estimator": self.use_reliability_estimator,
        }