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
    "mean_intensity",
    "std_intensity",
    "motion_energy",
    "motion_area_ratio",
    "motion_centroid_x",
    "motion_centroid_y",
    "motion_spread_x",
    "motion_spread_y",
]


def _resolve_usage_count(total_count: int, usage_ratio: float, name: str) -> int:
    if usage_ratio <= 0.0 or usage_ratio > 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1], got {usage_ratio}.")
    return max(1, int(round(total_count * usage_ratio)))


def _load_npy(path: Path, mmap_mode: str = "r") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed KTH Action array not found: {path}\n"
            "Run scripts/preprocess_kth_action.py first."
        )
    return np.load(path, mmap_mode=mmap_mode)


def _load_arrays(data_dir: str | Path, mmap_mode: str = "r") -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    images = _load_npy(data_dir / "images.npy", mmap_mode=mmap_mode)
    numbers = _load_npy(data_dir / "numbers.npy", mmap_mode=mmap_mode)
    if len(images) != len(numbers):
        raise ValueError(
            f"images.npy and numbers.npy have different sequence counts: {len(images)} vs {len(numbers)}."
        )
    if images.ndim != 5:
        raise ValueError(f"images.npy must have shape [N, T, H, W, C], got {images.shape}.")
    if numbers.ndim != 3:
        raise ValueError(f"numbers.npy must have shape [N, T, D], got {numbers.shape}.")
    if images.shape[1] != numbers.shape[1]:
        raise ValueError(
            f"Image and numerical sequence lengths differ: {images.shape[1]} vs {numbers.shape[1]}."
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


def _split_sequence_ranges(num_sequences: int) -> dict[str, tuple[int, int]]:
    prior_end = int(round(num_sequences * PRIOR_RATIO))
    test_start = int(round(num_sequences * (1.0 - TEST_RATIO)))
    val_start = int(round(num_sequences * (1.0 - TEST_RATIO - VAL_RATIO)))
    return {
        "prior_train": (0, prior_end),
        "gemini_train": (prior_end, val_start),
        "val": (val_start, test_start),
        "test": (test_start, num_sequences),
        "norm_train": (0, val_start),
        "train_all": (0, val_start),
    }


def _window_index_for_sequence_range(
    split: str,
    num_sequences: int,
    sequence_len: int,
    input_len_img: int,
    input_len_num: int,
    pred_len_img: int,
    pred_len_num: int,
    window_stride: int,
    test_usage_ratio: float = 1.0,
) -> list[tuple[int, int]]:
    ranges = _split_sequence_ranges(num_sequences)
    if split not in ranges:
        raise ValueError(f"Unknown split: {split}")
    sequence_start, sequence_end = ranges[split]
    required_len = max(input_len_img + pred_len_img, input_len_num + pred_len_num)
    if sequence_len < required_len:
        raise ValueError(
            f"KTH Action sequence is too short for requested lengths. "
            f"sequence_len={sequence_len}, required_len={required_len}."
        )
    if window_stride <= 0:
        raise ValueError(f"window_stride must be positive, got {window_stride}.")

    index: list[tuple[int, int]] = []
    for sequence_index in range(sequence_start, sequence_end):
        for start in range(0, sequence_len - required_len + 1, window_stride):
            index.append((sequence_index, start))

    if split == "test":
        count = _resolve_usage_count(len(index), test_usage_ratio, "test_usage_ratio")
        index = index[:count]
    return index


def build_normalizer(
    data_dir: str | Path,
    img_norm_type: str = "scale_01",
    num_norm_type: str = "zscore",
    img_value_scale: float = 255.0,
    data_usage_ratio: float = 1.0,
    num_vars: int = 8,
) -> MultimodalNormalizer:
    images, numbers = _load_arrays(data_dir=data_dir, mmap_mode="r")
    total_sequences = _resolve_usage_count(len(images), data_usage_ratio, "data_usage_ratio")
    images = images[:total_sequences]
    numbers = numbers[:total_sequences, :, :num_vars]
    _validate_num_vars(num_vars, numbers.shape[-1])

    _, norm_end = _split_sequence_ranges(total_sequences)["norm_train"]
    image_normalizer = ArrayNormalizer.fit(
        arrays=[images[:norm_end]],
        norm_type=img_norm_type,
        value_scale=img_value_scale,
        reduce_axes=None,
    )
    numerical_normalizer = NumericalNormalizer.from_arrays(
        x_num=numbers[:norm_end],
        y_num=numbers[:norm_end],
        norm_type=num_norm_type,
    )
    return MultimodalNormalizer(image=image_normalizer, numerical=numerical_normalizer)


class KTHActionDataset(Dataset):
    """KTH Action grayscale video + video-derived motion-statistic forecasting dataset."""

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
        num_vars: int = 8,
        img_height: int = 120,
        img_width: int = 120,
        img_channels: int = 1,
        window_stride: int = 5,
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
                f"Processed image channels do not match run.py settings. "
                f"Expected C={img_channels}, but got C={images.shape[-1]}."
            )
        total_sequences = _resolve_usage_count(len(images), data_usage_ratio, "data_usage_ratio")
        self.images = images[:total_sequences]
        self.numbers = numbers[:total_sequences, :, :num_vars]
        _validate_num_vars(num_vars, self.numbers.shape[-1])
        self.window_index = _window_index_for_sequence_range(
            split=split,
            num_sequences=total_sequences,
            sequence_len=self.images.shape[1],
            input_len_img=input_len_img,
            input_len_num=input_len_num,
            pred_len_img=pred_len_img,
            pred_len_num=pred_len_num,
            window_stride=window_stride,
            test_usage_ratio=test_usage_ratio,
        )

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sequence_index, start = self.window_index[index]

        x_img_raw = torch.from_numpy(
            np.array(self.images[sequence_index, start : start + self.input_len_img], copy=True)
        ).float()
        y_img_raw = torch.from_numpy(
            np.array(
                self.images[
                    sequence_index,
                    start + self.input_len_img : start + self.input_len_img + self.pred_len_img,
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
            np.array(self.numbers[sequence_index, start : start + self.input_len_num], copy=True)
        ).float()
        y_num_raw = torch.from_numpy(
            np.array(
                self.numbers[
                    sequence_index,
                    start + self.input_len_num : start + self.input_len_num + self.pred_len_num,
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
    img_norm_type: str = "scale_01",
    num_norm_type: str = "zscore",
    img_value_scale: float = 255.0,
    data_usage_ratio: float = 1.0,
    test_usage_ratio: float = 1.0,
    num_vars: int = 8,
    input_len_img: int = 15,
    input_len_num: int = 15,
    pred_len_img: int = 15,
    pred_len_num: int = 15,
    img_height: int = 120,
    img_width: int = 120,
    img_channels: int = 1,
    window_stride: int = 5,
) -> Tuple[Dict[str, DataLoader], MultimodalNormalizer]:
    normalizer = build_normalizer(
        data_dir=data_dir,
        img_norm_type=img_norm_type,
        num_norm_type=num_norm_type,
        img_value_scale=img_value_scale,
        data_usage_ratio=data_usage_ratio,
        num_vars=num_vars,
    )
    loaders: Dict[str, DataLoader] = {}
    for split in ["prior_train", "gemini_train", "val", "test"]:
        dataset = KTHActionDataset(
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