from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .normalizer import ArrayNormalizer, MultimodalNormalizer, NumericalNormalizer

PRIOR_RATIO = 0.10
VAL_RATIO = 0.10
TEST_RATIO = 0.10
DEFAULT_NUM_FEATURE_NAMES = [
    "wind_speed",
    "station_precipitation",
    "relative_humidity",
    "dew_point_temperature",
    "temperature",
    "sea_level_pressure",
]


def _resolve_usage_count(total_count: int, usage_ratio: float, name: str) -> int:
    if usage_ratio <= 0.0 or usage_ratio > 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1], got {usage_ratio}.")
    return max(1, int(round(total_count * usage_ratio)))


def _load_npy(path: Path, mmap_mode: str = "r") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Converted MeteoNet array not found: {path}\n"
            "Run scripts/preprocess_meteonet_nw.py first."
        )
    return np.load(path, mmap_mode=mmap_mode)


def _load_arrays(data_dir: str | Path, mmap_mode: str = "r") -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    images = _load_npy(data_dir / "images.npy", mmap_mode=mmap_mode)
    numbers = _load_npy(data_dir / "numbers.npy", mmap_mode=mmap_mode)
    if len(images) != len(numbers):
        raise ValueError(
            f"images.npy and numbers.npy have different lengths: {len(images)} vs {len(numbers)}."
        )
    return images, numbers


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


def _split_time_ranges(num_timesteps: int) -> dict[str, tuple[int, int]]:
    prior_end = int(round(num_timesteps * PRIOR_RATIO))
    test_start = int(round(num_timesteps * (1.0 - TEST_RATIO)))
    val_start = int(round(num_timesteps * (1.0 - TEST_RATIO - VAL_RATIO)))

    return {
        "prior_train": (0, prior_end),
        "gemini_train": (prior_end, val_start),
        "val": (val_start, test_start),
        "test": (test_start, num_timesteps),
        "norm_train": (0, val_start),
        "train_all": (0, val_start),
    }


def _window_starts_for_time_range(
    split: str,
    num_timesteps: int,
    input_len_img: int,
    input_len_num: int,
    pred_len_img: int,
    pred_len_num: int,
    test_usage_ratio: float = 1.0,
) -> np.ndarray:
    ranges = _split_time_ranges(num_timesteps)
    if split not in ranges:
        raise ValueError(f"Unknown split: {split}")

    start_time, end_time = ranges[split]
    required_img = input_len_img + pred_len_img
    required_num = input_len_num + pred_len_num
    required_len = max(required_img, required_num)

    max_start = end_time - required_len
    if max_start < start_time:
        raise ValueError(
            f"Split '{split}' is too short for the requested sequence lengths. "
            f"time_range=({start_time}, {end_time}), required_len={required_len}."
        )

    starts = np.arange(start_time, max_start + 1, dtype=np.int64)
    if split == "test":
        count = _resolve_usage_count(len(starts), test_usage_ratio, "test_usage_ratio")
        starts = starts[:count]
    return starts


def build_normalizer(
    data_dir: str | Path,
    img_norm_type: str = "log1p_zscore",
    num_norm_type: str = "zscore",
    img_value_scale: float = 1.0,
    data_usage_ratio: float = 1.0,
    num_vars: int = 6,
) -> MultimodalNormalizer:
    images, numbers = _load_arrays(data_dir=data_dir, mmap_mode="r")
    total_timesteps = _resolve_usage_count(len(images), data_usage_ratio, "data_usage_ratio")
    images = images[:total_timesteps]
    numbers = numbers[:total_timesteps, :num_vars]
    _validate_num_vars(num_vars, numbers.shape[-1])

    ranges = _split_time_ranges(total_timesteps)
    _, norm_end = ranges["norm_train"]

    image_normalizer = ArrayNormalizer.fit(
        arrays=[images[:norm_end]],
        norm_type=img_norm_type,
        value_scale=img_value_scale,
        reduce_axes=None,
    )
    numerical_normalizer = NumericalNormalizer.from_arrays(
        x_num=numbers[:norm_end][None, ...],
        y_num=numbers[:norm_end][None, ...],
        norm_type=num_norm_type,
    )

    return MultimodalNormalizer(image=image_normalizer, numerical=numerical_normalizer)


class MeteoNetNWMultimodalDataset(Dataset):
    """MeteoNet NW hourly multimodal forecasting dataset.

    Tensor layout:
        x_img/y_img: [T, H, W, 1], hourly radar rainfall maps.
        x_num/y_num: [T, D], regional ground-station meteorological variables.
    """

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
        num_vars: int = 6,
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
        self.metadata = load_metadata(self.data_dir)
        self.num_feature_names = get_num_feature_names(num_vars, self.metadata)

        images, numbers = _load_arrays(self.data_dir, mmap_mode="r")
        total_timesteps = _resolve_usage_count(len(images), data_usage_ratio, "data_usage_ratio")
        self.images = images[:total_timesteps]
        self.numbers = numbers[:total_timesteps, :num_vars]
        _validate_num_vars(num_vars, self.numbers.shape[-1])

        self.starts = _window_starts_for_time_range(
            split=split,
            num_timesteps=total_timesteps,
            input_len_img=input_len_img,
            input_len_num=input_len_num,
            pred_len_img=pred_len_img,
            pred_len_num=pred_len_num,
            test_usage_ratio=test_usage_ratio,
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = int(self.starts[index])

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

        x_img = self.normalizer.image.normalize(x_img_raw)
        y_img = self.normalizer.image.normalize(y_img_raw)
        x_num = self.normalizer.numerical.normalize(x_num_raw)
        y_num = self.normalizer.numerical.normalize(y_num_raw)

        return {
            "x_img": x_img,
            "y_img": y_img,
            "x_img_raw": x_img_raw,
            "y_img_raw": y_img_raw,
            "x_num": x_num,
            "y_num": y_num,
            "x_num_raw": x_num_raw,
            "y_num_raw": y_num_raw,
        }


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    img_norm_type: str = "log1p_zscore",
    num_norm_type: str = "zscore",
    img_value_scale: float = 1.0,
    data_usage_ratio: float = 1.0,
    test_usage_ratio: float = 1.0,
    num_vars: int = 6,
    input_len_img: int = 12,
    input_len_num: int = 12,
    pred_len_img: int = 12,
    pred_len_num: int = 12,
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
        dataset = MeteoNetNWMultimodalDataset(
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