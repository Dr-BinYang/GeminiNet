from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .normalizer import ArrayNormalizer, MultimodalNormalizer, NumericalNormalizer

TRAIN_PRIOR_RATIO = 0.10
TRAIN_VAL_RATIO = 0.10
DEFAULT_NUM_FEATURE_NAMES = [
    "ee_x",
    "ee_y",
    "ee_z",
    "action_1",
    "action_2",
    "action_3",
    "action_4",
]


def _resolve_usage_count(
    total_count: int,
    usage_ratio: float,
    name: str,
) -> int:
    if usage_ratio <= 0.0 or usage_ratio > 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1], got {usage_ratio}.")

    return max(1, int(round(total_count * usage_ratio)))


def _validate_train_count(train_count: int) -> None:
    prior_start, prior_end = split_range("prior_train", train_count)
    gemini_start, gemini_end = split_range("gemini_train", train_count)
    val_start, val_end = split_range("val", train_count)

    if prior_end - prior_start <= 0 or gemini_end - gemini_start <= 0 or val_end - val_start <= 0:
        raise ValueError(
            "data_usage_ratio is too small for the BAIR split protocol. "
            f"Effective train samples: {train_count}. "
            "Increase data_usage_ratio."
        )


def _load_npy(path: Path, mmap_mode: str = "r") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Converted BAIR array not found: {path}\n"
            "Run scripts/preprocess_bair_robot_pushing.py first."
        )
    return np.load(path, mmap_mode=mmap_mode)


def _load_split_arrays(
    data_dir: str | Path,
    split: str,
    mmap_mode: str = "r",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)

    x_img = _load_npy(data_dir / f"{split}_x_img.npy", mmap_mode=mmap_mode)
    y_img = _load_npy(data_dir / f"{split}_y_img.npy", mmap_mode=mmap_mode)
    x_num = _load_npy(data_dir / f"{split}_x_num.npy", mmap_mode=mmap_mode)
    y_num = _load_npy(data_dir / f"{split}_y_num.npy", mmap_mode=mmap_mode)

    return x_img, y_img, x_num, y_num


def _validate_num_vars(num_vars: int, available_num_vars: int) -> None:
    if num_vars <= 0:
        raise ValueError(f"num_vars must be positive, got {num_vars}.")

    if num_vars > available_num_vars:
        raise ValueError(
            f"num_vars={num_vars} exceeds available numerical dimensions " f"{available_num_vars}."
        )


def _select_num_features(array: np.ndarray, num_vars: int) -> np.ndarray:
    """Select the first num_vars dimensions as the numerical modality."""
    _validate_num_vars(
        num_vars=num_vars,
        available_num_vars=array.shape[-1],
    )
    return array[..., :num_vars]


def get_num_feature_names(num_vars: int) -> list[str]:
    """Return human-readable names for selected numerical dimensions."""
    if num_vars <= len(DEFAULT_NUM_FEATURE_NAMES):
        return DEFAULT_NUM_FEATURE_NAMES[:num_vars]

    extra_names = [f"var_{index}" for index in range(len(DEFAULT_NUM_FEATURE_NAMES), num_vars)]
    return DEFAULT_NUM_FEATURE_NAMES + extra_names


def load_metadata(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "metadata.json"
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_range(split: str, n_train: int, n_test: int | None = None) -> Tuple[int, int]:
    """Return index range for the BAIR experiment protocol.

    BAIR already provides an official test split. We therefore split only the
    official train trajectories into prior_train, gemini_train and val:

        prior_train:
            first 10% of official train data.

        gemini_train:
            middle 80% of official train data.

        val:
            final 10% of official train data.

        test:
            full official test data.

        norm_train:
            prior_train + gemini_train, excluding val and test.
    """
    idx_prior_end = int(round(n_train * TRAIN_PRIOR_RATIO))
    idx_val_start = int(round(n_train * (1.0 - TRAIN_VAL_RATIO)))

    if split == "prior_train":
        return 0, idx_prior_end

    if split == "gemini_train":
        return idx_prior_end, idx_val_start

    if split == "val":
        return idx_val_start, n_train

    if split == "norm_train":
        return 0, idx_val_start

    if split == "test":
        if n_test is None:
            raise ValueError("n_test is required for split='test'.")
        return 0, n_test

    if split == "train_all":
        return 0, n_train

    raise ValueError(f"Unknown split: {split}")


def build_normalizer(
    data_dir: str | Path,
    img_norm_type: str = "scale_01",
    num_norm_type: str = "zscore",
    img_value_scale: float = 255.0,
    data_usage_ratio: float = 1.0,
    num_vars: int = 5,
) -> MultimodalNormalizer:
    """Compute BAIR normalizers from train-only data.

    Image pixels are normally uint8 RGB, so scale_01 is the recommended image
    normalization. Numerical statistics are fitted from prior_train +
    gemini_train only; validation and official test data are excluded.
    """
    x_img_train, y_img_train, x_num_train, y_num_train = _load_split_arrays(
        data_dir=data_dir,
        split="train",
        mmap_mode="r",
    )

    train_count = _resolve_usage_count(
        total_count=len(x_num_train),
        usage_ratio=data_usage_ratio,
        name="data_usage_ratio",
    )
    _validate_train_count(train_count)

    _, norm_end = split_range("norm_train", train_count)

    if img_norm_type not in ["none", "scale_01"]:
        raise ValueError(
            "BAIR RGB arrays are large; this data provider currently supports "
            "img_norm_type='none' or 'scale_01'. Use scale_01 for uint8 pixels."
        )

    image_normalizer = ArrayNormalizer.fit(
        arrays=[x_img_train[:1], y_img_train[:1]],
        norm_type=img_norm_type,
        value_scale=img_value_scale,
        reduce_axes=None,
    )

    x_num_for_stats = _select_num_features(x_num_train[:norm_end], num_vars)
    y_num_for_stats = _select_num_features(y_num_train[:norm_end], num_vars)

    numerical_normalizer = NumericalNormalizer.from_arrays(
        x_num=x_num_for_stats,
        y_num=y_num_for_stats,
        norm_type=num_norm_type,
    )

    return MultimodalNormalizer(
        image=image_normalizer,
        numerical=numerical_normalizer,
    )


class BairRobotPushingMultimodalDataset(Dataset):
    """BAIR Robot Pushing Small dataset for GeminiNet.

    Internal tensor layout:
        x_img/y_img: [T, 64, 64, 3]
        x_num/y_num: [T, D], where D is usually 7
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        normalizer: MultimodalNormalizer,
        data_usage_ratio: float = 1.0,
        test_usage_ratio: float = 1.0,
        num_vars: int = 5,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.normalizer = normalizer
        self.num_vars = num_vars
        self.metadata = load_metadata(self.data_dir)
        self.num_feature_names = get_num_feature_names(num_vars)

        train_arrays = _load_split_arrays(
            data_dir=self.data_dir,
            split="train",
            mmap_mode="r",
        )
        test_arrays = _load_split_arrays(
            data_dir=self.data_dir,
            split="test",
            mmap_mode="r",
        )

        n_train = _resolve_usage_count(
            total_count=len(train_arrays[0]),
            usage_ratio=data_usage_ratio,
            name="data_usage_ratio",
        )
        _validate_train_count(n_train)
        n_test = _resolve_usage_count(
            total_count=len(test_arrays[0]),
            usage_ratio=test_usage_ratio,
            name="test_usage_ratio",
        )

        if split == "test":
            arrays = tuple(array[:n_test] for array in test_arrays)
            start, end = split_range(split, n_train=n_train, n_test=n_test)
        else:
            arrays = tuple(array[:n_train] for array in train_arrays)
            start, end = split_range(split, n_train=n_train, n_test=n_test)

        self.x_img = arrays[0][start:end]
        self.y_img = arrays[1][start:end]
        self.x_num = _select_num_features(arrays[2][start:end], num_vars)
        self.y_num = _select_num_features(arrays[3][start:end], num_vars)

    def __len__(self) -> int:
        return len(self.x_img)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x_img_raw = torch.from_numpy(np.array(self.x_img[index], copy=True)).float()
        y_img_raw = torch.from_numpy(np.array(self.y_img[index], copy=True)).float()

        x_img = self.normalizer.image.normalize(x_img_raw)
        y_img = self.normalizer.image.normalize(y_img_raw)

        x_num_raw = torch.from_numpy(np.array(self.x_num[index], copy=True)).float()
        y_num_raw = torch.from_numpy(np.array(self.y_num[index], copy=True)).float()

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
    img_norm_type: str = "scale_01",
    num_norm_type: str = "zscore",
    img_value_scale: float = 255.0,
    data_usage_ratio: float = 1.0,
    test_usage_ratio: float = 1.0,
    num_vars: int = 5,
) -> Tuple[Dict[str, DataLoader], MultimodalNormalizer]:
    """Build BAIR dataloaders for all GeminiNet stages."""
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
        dataset = BairRobotPushingMultimodalDataset(
            data_dir=data_dir,
            split=split,
            normalizer=normalizer,
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