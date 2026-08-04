from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from data_provider import build_dataloaders, split_range
from data_provider.normalizer import ArrayNormalizer
from models import GeminiNet, ImageUNetPrior, NumericalTransformerPrior
from run import check_mode, parse_args
from utils.metrics import build_reliability_labels, gemininet_loss


def test_normalizer_round_trip() -> None:
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    for norm_type in [
        "none",
        "scale_01",
        "minmax",
        "zscore",
        "robust",
        "log1p_zscore",
    ]:
        normalizer = ArrayNormalizer(norm_type=norm_type).fit(data)
        restored = normalizer.inverse_transform(normalizer.transform(data))
        assert np.allclose(restored, data, atol=1e-4)


def test_default_configuration_is_valid() -> None:
    args = parse_args([])
    check_mode(args)
    assert args.img_norm_type == "scale_01"
    assert args.img_output_activation == "sigmoid"
    assert args.num_norm_type == "zscore"
    assert args.num_output_activation == "linear"
    assert args.input_len_img == 15
    assert args.input_len_num == 15
    assert args.pred_len_img == 15
    assert args.pred_len_num == 15


def test_log1p_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ArrayNormalizer("log1p_zscore").fit(np.array([-1.0, 0.0], dtype=np.float32))


def _write_split(path: Path, count: int) -> None:
    rng = np.random.default_rng(42 + count)
    np.savez(
        path,
        x_img=rng.integers(
            0,
            256,
            size=(count, 2, 1, 16, 16),
            dtype=np.uint8,
        ),
        y_img=rng.integers(
            0,
            256,
            size=(count, 2, 1, 16, 16),
            dtype=np.uint8,
        ),
        x_num=rng.normal(size=(count, 2, 3)).astype(np.float32),
        y_num=rng.normal(size=(count, 2, 3)).astype(np.float32),
    )


def test_official_files_remain_separate(tmp_path: Path) -> None:
    _write_split(tmp_path / "train.npz", 8)
    _write_split(tmp_path / "val.npz", 3)
    _write_split(tmp_path / "test.npz", 4)

    loaders, _ = build_dataloaders(
        data_dir=tmp_path,
        batch_size=2,
        prior_fraction=0.25,
    )

    assert len(loaders["prior_train"].dataset) == 2
    assert len(loaders["gemini_train"].dataset) == 6
    assert len(loaders["val"].dataset) == 3
    assert len(loaders["test"].dataset) == 4
    assert split_range(
        "gemini_train",
        8,
        prior_fraction=0.25,
        split_gap=1,
    ) == (3, 8)


def _build_small_model() -> GeminiNet:
    image_prior = ImageUNetPrior(
        input_len=2,
        pred_len=2,
        img_channels=1,
        base_channels=4,
        dropout=0.0,
    )
    num_prior = NumericalTransformerPrior(
        input_len=2,
        pred_len=2,
        num_vars=3,
        d_model=8,
        nhead=2,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )

    return GeminiNet(
        image_prior=image_prior,
        num_prior=num_prior,
        input_len_img=2,
        input_len_num=2,
        pred_len_img=2,
        pred_len_num=2,
        img_size=32,
        img_channels=1,
        num_vars=3,
        d_img=8,
        d_num=8,
        d_shared=8,
        nhead=2,
        dropout=0.0,
    )


def test_gemininet_forward_backward_uses_specific_memory() -> None:
    model = _build_small_model()
    x_img = torch.rand(2, 2, 32, 32, 1)
    x_num = torch.randn(2, 2, 3)
    y_img = torch.rand(2, 2, 32, 32, 1)
    y_num = torch.randn(2, 2, 3)

    outputs = model(x_img, x_num)
    loss, _ = gemininet_loss(outputs, y_img, y_num)
    loss.backward()

    assert outputs["y_img"].shape == y_img.shape
    assert outputs["y_num"].shape == y_num.shape
    assert model.multimodal_interaction.img_output[0].weight.grad is not None
    assert model.tendency_generator.image_prior.temporal_stem[0].weight.grad is None


def test_image_prior_supports_non_power_of_two_size() -> None:
    model = ImageUNetPrior(
        input_len=2,
        pred_len=2,
        img_channels=1,
        base_channels=4,
        dropout=0.0,
    )
    output = model(torch.rand(1, 2, 30, 34, 1))
    assert output.shape == (1, 2, 30, 34, 1)


def test_relative_reliability_is_batch_independent() -> None:
    p_img = torch.tensor(
        [
            [[[[0.0]]], [[[1.0]]]],
            [[[[2.0]]], [[[4.0]]]],
        ]
    )
    y_img = torch.zeros_like(p_img)
    p_num = torch.tensor(
        [
            [[0.0], [1.0]],
            [[2.0], [4.0]],
        ]
    )
    y_num = torch.zeros_like(p_num)

    full_img, full_num = build_reliability_labels(
        p_img,
        y_img,
        p_num,
        y_num,
        mode="relative",
    )
    single_img, single_num = build_reliability_labels(
        p_img[:1],
        y_img[:1],
        p_num[:1],
        y_num[:1],
        mode="relative",
    )

    assert torch.allclose(full_img[:1], single_img)
    assert torch.allclose(full_num[:1], single_num)