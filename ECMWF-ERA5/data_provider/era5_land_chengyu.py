from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .normalizer import ArrayNormalizer, MultimodalNormalizer, NumericalNormalizer

PRIOR_RATIO = 0.10
VAL_RATIO = 0.10
TEST_RATIO = 0.10

DEFAULT_NUM_FEATURE_NAMES = [
    "t2m_mean_c",
    "d2m_mean_c",
    "u10_mean_ms",
    "v10_mean_ms",
    "wind10_mean_ms",
    "sp_mean_hpa",
    "swvl1_mean_m3m3",
    "stl1_mean_c",
    "skt_mean_c",
]


def _resolve_usage_count(total_count: int, usage_ratio: float, name: str) -> int:
    if usage_ratio <= 0.0 or usage_ratio > 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1], got {usage_ratio}.")
    return max(1, int(round(total_count * usage_ratio)))


def _load_npy(path: Path, mmap_mode: str = "r") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed ERA5-Land array not found: {path}\n"
            "Run scripts/preprocess_era5_land_chengyu.py first."
        )
    return np.load(path, mmap_mode=mmap_mode)


def _load_arrays(data_dir: str | Path, mmap_mode: str = "r") -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    images = _load_npy(data_dir / "images.npy", mmap_mode=mmap_mode)
    numbers = _load_npy(data_dir / "numbers.npy", mmap_mode=mmap_mode)
    if images.ndim != 4:
        raise ValueError(f"images.npy must have shape [T, H, W, C], got {images.shape}.")
    if numbers.ndim != 2:
        raise ValueError(f"numbers.npy must have shape [T, D], got {numbers.shape}.")
    if len(images) != len(numbers):
        raise ValueError(
            f"images.npy and numbers.npy have different time lengths: {len(images)} vs {len(numbers)}."
        )
    return images, numbers


def _resize_nhwc_sequence(images: torch.Tensor, img_height: int, img_width: int) -> torch.Tensor:
    """Resize an image sequence from [T, H, W, C] to [T, img_height, img_width, C]."""
    if images.shape[1] == img_height and images.shape[2] == img_width:
        return images

    time_steps, _height, _width, channels = images.shape
    images_nchw = images.permute(0, 3, 1, 2).contiguous()
    images_nchw = F.interpolate(
        images_nchw,
        size=(img_height, img_width),
        mode="bilinear",
        align_corners=False,
    )
    return (
        images_nchw.permute(0, 2, 3, 1)
        .reshape(time_steps, img_height, img_width, channels)
        .contiguous()
    )


def load_metadata(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_num_feature_names(num_vars: int, metadata: dict | None = None) -> list[str]:
    names = None
    if metadata is not None:
        names = metadata.get("num_feature_names")
    if not names:
        names = DEFAULT_NUM_FEATURE_NAMES
    if num_vars <= len(names):
        return list(names[:num_vars])
    return list(names) + [f"var_{index}" for index in range(len(names), num_vars)]


def _validate_num_vars(num_vars: int, available_num_vars: int) -> None:
    if num_vars <= 0:
        raise ValueError(f"num_vars must be positive, got {num_vars}.")
    if num_vars > available_num_vars:
        raise ValueError(
            f"num_vars={num_vars} exceeds available numerical dimensions {available_num_vars}."
        )


def _split_window_ranges(num_windows: int) -> dict[str, tuple[int, int]]:
    prior_end = int(round(num_windows * PRIOR_RATIO))
    test_start = int(round(num_windows * (1.0 - TEST_RATIO)))
    val_start = int(round(num_windows * (1.0 - TEST_RATIO - VAL_RATIO)))
    return {
        "prior_train": (0, prior_end),
        "gemini_train": (prior_end, val_start),
        "val": (val_start, test_start),
        "test": (test_start, num_windows),
        "norm_train": (0, val_start),
        "train_all": (0, val_start),
    }


def _window_starts(
    time_steps: int,
    input_len_img: int,
    input_len_num: int,
    pred_len_img: int,
    pred_len_num: int,
    window_stride: int,
) -> list[int]:
    required_len = max(input_len_img + pred_len_img, input_len_num + pred_len_num)
    if time_steps < required_len:
        raise ValueError(
            f"ERA5-Land time series is too short: time_steps={time_steps}, required_len={required_len}."
        )
    if window_stride <= 0:
        raise ValueError(f"window_stride must be positive, got {window_stride}.")
    return list(range(0, time_steps - required_len + 1, window_stride))


def _window_index_for_split(
    split: str,
    starts: list[int],
    test_usage_ratio: float = 1.0,
) -> list[int]:
    ranges = _split_window_ranges(len(starts))
    if split not in ranges:
        raise ValueError(f"Unknown split: {split}")
    start, end = ranges[split]
    index = starts[start:end]
    if split == "test":
        count = _resolve_usage_count(len(index), test_usage_ratio, "test_usage_ratio")
        index = index[:count]
    return index


def build_normalizer(
    data_dir: str | Path,
    img_norm_type: str = "zscore",
    num_norm_type: str = "zscore",
    img_value_scale: float = 1.0,
    data_usage_ratio: float = 1.0,
    num_vars: int = 9,
    input_len_img: int = 12,
    input_len_num: int = 12,
    pred_len_img: int = 12,
    pred_len_num: int = 12,
    window_stride: int = 1,
) -> MultimodalNormalizer:
    images, numbers = _load_arrays(data_dir=data_dir, mmap_mode="r")
    total_time_steps = _resolve_usage_count(len(images), data_usage_ratio, "data_usage_ratio")
    images = images[:total_time_steps]
    numbers = numbers[:total_time_steps, :num_vars]
    _validate_num_vars(num_vars, numbers.shape[-1])

    starts = _window_starts(
        time_steps=total_time_steps,
        input_len_img=input_len_img,
        input_len_num=input_len_num,
        pred_len_img=pred_len_img,
        pred_len_num=pred_len_num,
        window_stride=window_stride,
    )
    _, norm_window_end = _split_window_ranges(len(starts))["norm_train"]
    required_len = max(input_len_img + pred_len_img, input_len_num + pred_len_num)
    norm_time_end = min(total_time_steps, starts[max(0, norm_window_end - 1)] + required_len)

    image_normalizer = ArrayNormalizer.fit(
        arrays=[images[:norm_time_end]],
        norm_type=img_norm_type,
        value_scale=img_value_scale,
        reduce_axes=None,
    )
    numerical_normalizer = NumericalNormalizer.fit_from_arrays(
        x_num=numbers[:norm_time_end][None, :, :],
        y_num=numbers[:norm_time_end][None, :, :],
        norm_type=num_norm_type,
    )
    return MultimodalNormalizer(image=image_normalizer, numerical=numerical_normalizer)


class ERA5LandChengYuDataset(Dataset):
    """ERA5-Land Chengdu-Chongqing regional field + regional-mean state forecasting dataset."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        normalizer: MultimodalNormalizer,
        input_len_img: int,
        input_len_num: int,
        pred_len_img: int,
        pred_len_num: int,
        data_usage_ratio: float = 1.0,
        test_usage_ratio: float = 1.0,
        num_vars: int = 9,
        img_height: int = 101,
        img_width: int = 101,
        img_channels: int = 1,
        window_stride: int = 1,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalizer = normalizer
        self.input_len_img = input_len_img
        self.input_len_num = input_len_num
        self.pred_len_img = pred_len_img
        self.pred_len_num = pred_len_num
        self.num_vars = num_vars
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.metadata = load_metadata(self.data_dir)
        self.num_feature_names = get_num_feature_names(num_vars, self.metadata)

        images, numbers = _load_arrays(self.data_dir, mmap_mode="r")
        if images.shape[-1] != img_channels:
            raise ValueError(
                f"Processed image channels do not match run.py settings. Expected C={img_channels}, got C={images.shape[-1]}."
            )
        total_time_steps = _resolve_usage_count(len(images), data_usage_ratio, "data_usage_ratio")
        self.images = images[:total_time_steps]
        self.numbers = numbers[:total_time_steps, :num_vars]
        _validate_num_vars(num_vars, self.numbers.shape[-1])
        starts = _window_starts(
            time_steps=total_time_steps,
            input_len_img=input_len_img,
            input_len_num=input_len_num,
            pred_len_img=pred_len_img,
            pred_len_num=pred_len_num,
            window_stride=window_stride,
        )
        self.window_index = _window_index_for_split(
            split=split,
            starts=starts,
            test_usage_ratio=test_usage_ratio,
        )

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = self.window_index[index]
        x_img_raw = torch.from_numpy(
            np.array(self.images[start : start + self.input_len_img], copy=True)
        ).float()
        y_img_raw = torch.from_numpy(
            np.array(
                self.images[
                    start + self.input_len_img : start + self.input_len_img + self.pred_len_img
                ],
                copy=True,
            )
        ).float()
        x_img_raw = _resize_nhwc_sequence(
            x_img_raw, img_height=self.img_height, img_width=self.img_width
        )
        y_img_raw = _resize_nhwc_sequence(
            y_img_raw, img_height=self.img_height, img_width=self.img_width
        )

        x_num_raw = torch.from_numpy(
            np.array(self.numbers[start : start + self.input_len_num], copy=True)
        ).float()
        y_num_raw = torch.from_numpy(
            np.array(
                self.numbers[
                    start + self.input_len_num : start + self.input_len_num + self.pred_len_num
                ],
                copy=True,
            )
        ).float()

        return {
            "x_img": self.normalizer.image.normalize(x_img_raw),
            "y_img": self.normalizer.image.normalize(y_img_raw),
            "x_img_raw": x_img_raw,
            "y_img_raw": y_img_raw,
            "x_num": self.normalizer.numerical.normalize(x_num_raw),
            "y_num": self.normalizer.numerical.normalize(y_num_raw),
            "x_num_raw": x_num_raw,
            "y_num_raw": y_num_raw,
        }


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    img_norm_type: str = "zscore",
    num_norm_type: str = "zscore",
    img_value_scale: float = 1.0,
    data_usage_ratio: float = 1.0,
    test_usage_ratio: float = 1.0,
    num_vars: int = 9,
    input_len_img: int = 12,
    input_len_num: int = 12,
    pred_len_img: int = 12,
    pred_len_num: int = 12,
    img_height: int = 101,
    img_width: int = 101,
    img_channels: int = 1,
    window_stride: int = 1,
) -> Tuple[Dict[str, DataLoader], MultimodalNormalizer]:
    normalizer = build_normalizer(
        data_dir=data_dir,
        img_norm_type=img_norm_type,
        num_norm_type=num_norm_type,
        img_value_scale=img_value_scale,
        data_usage_ratio=data_usage_ratio,
        num_vars=num_vars,
        input_len_img=input_len_img,
        input_len_num=input_len_num,
        pred_len_img=pred_len_img,
        pred_len_num=pred_len_num,
        window_stride=window_stride,
    )
    loaders: Dict[str, DataLoader] = {}
    for split in ["prior_train", "gemini_train", "val", "test"]:
        dataset = ERA5LandChengYuDataset(
            data_dir=data_dir,
            split=split,
            normalizer=normalizer,
            input_len_img=input_len_img,
            input_len_num=input_len_num,
            pred_len_img=pred_len_img,
            pred_len_num=pred_len_num,
            data_usage_ratio=data_usage_ratio,
            test_usage_ratio=test_usage_ratio,
            num_vars=num_vars,
            img_height=img_height,
            img_width=img_width,
            img_channels=img_channels,
            window_stride=window_stride,
        )
        loaders[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split in ["prior_train", "gemini_train"]),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return loaders, normalizer