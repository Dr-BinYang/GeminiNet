from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .normalizer import ArrayNormalizer

ArrayTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _to_nthwc(array: np.ndarray, name: str) -> np.ndarray:
    """Convert supported image layouts to [N, T, H, W, C]."""
    if array.ndim == 4:
        return array[..., np.newaxis].astype(np.float32)

    if array.ndim != 5:
        raise ValueError(
            f"{name} must have shape [N,T,C,H,W], [N,T,H,W,C], " f"or [N,T,H,W], got {array.shape}."
        )

    channel_first = array.shape[2] <= 4 or (
        array.shape[2] < array.shape[3] and array.shape[2] < array.shape[4]
    )
    channel_last = array.shape[-1] <= 4 or (
        array.shape[-1] < array.shape[2] and array.shape[-1] < array.shape[3]
    )

    if channel_first and not channel_last:
        array = np.transpose(array, (0, 1, 3, 4, 2))
    elif channel_first and channel_last:
        raise ValueError(f"Ambiguous channel axis for {name} with shape {array.shape}.")
    elif not channel_last:
        raise ValueError(f"Cannot infer the channel axis for {name} with shape " f"{array.shape}.")

    return array.astype(np.float32)


def _load_npz(path: Path) -> ArrayTuple:
    """Load one dataset file and normalize array layouts only."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"x_img", "y_img", "x_num", "y_num"}
        missing = required.difference(data.files)

        if missing:
            raise KeyError(f"Missing arrays in {path}: {sorted(missing)}")

        x_img = _to_nthwc(data["x_img"], name=f"{path.name}:x_img")
        y_img = _to_nthwc(data["y_img"], name=f"{path.name}:y_img")
        x_num = data["x_num"].astype(np.float32)
        y_num = data["y_num"].astype(np.float32)

    sample_counts = {
        len(x_img),
        len(y_img),
        len(x_num),
        len(y_num),
    }
    if len(sample_counts) != 1:
        raise ValueError(f"Inconsistent sample counts in {path}.")

    return x_img, y_img, x_num, y_num


def load_dataset_files(
    data_dir: str | Path,
) -> Dict[str, ArrayTuple]:
    """Load official train/validation/test files without recombining them."""
    data_dir = Path(data_dir)

    return {
        "train": _load_npz(data_dir / "train.npz"),
        "val": _load_npz(data_dir / "val.npz"),
        "test": _load_npz(data_dir / "test.npz"),
    }


def split_range(
    split: str,
    n_total: int,
    prior_fraction: float = 0.125,
    split_gap: int = 0,
) -> Tuple[int, int]:
    """Return an ordered range inside the official training file.

    The default 0.125 gives the prior models 1/8 of the official training
    samples and GeminiNet the remaining 7/8. Validation and test samples are
    never mixed into either training range.
    """
    if not 0.0 < prior_fraction < 1.0:
        raise ValueError("prior_fraction must be between 0 and 1.")
    if split_gap < 0:
        raise ValueError("split_gap must be non-negative.")

    prior_end = int(round(n_total * prior_fraction))
    prior_end = min(max(prior_end, 1), max(1, n_total - 1))
    gemini_start = prior_end + split_gap

    if gemini_start >= n_total:
        raise ValueError("prior_fraction and split_gap leave no GeminiNet samples.")

    if split == "prior_train":
        return 0, prior_end

    if split == "gemini_train":
        return gemini_start, n_total

    if split in {"norm_train", "train", "all"}:
        return 0, n_total

    raise ValueError(
        f"Unknown training split: {split}. Validation and test use their "
        "official files directly."
    )


def build_normalizers(
    data_dir: str | Path,
    img_norm_type: str = "scale_01",
    num_norm_type: str = "zscore",
    train_arrays: ArrayTuple | None = None,
) -> Dict[str, ArrayNormalizer]:
    """Fit separate modality normalizers on the official training file."""
    if train_arrays is None:
        train_arrays = load_dataset_files(data_dir)["train"]

    x_img, y_img, x_num, y_num = train_arrays

    return {
        "image": ArrayNormalizer(norm_type=img_norm_type).fit([x_img, y_img]),
        "numerical": ArrayNormalizer(norm_type=num_norm_type).fit([x_num, y_num]),
    }


def normalizers_from_dict(
    data: dict,
) -> Dict[str, ArrayNormalizer]:
    """Restore both modality normalizers from serialized metadata."""
    required = {"image", "numerical"}
    missing = required.difference(data)

    if missing:
        raise KeyError(f"Normalizer metadata is missing: {sorted(missing)}")

    return {name: ArrayNormalizer.from_dict(data[name]) for name in ["image", "numerical"]}


def build_normalizer(
    data_dir: str | Path,
    num_norm_type: str = "zscore",
) -> ArrayNormalizer:
    """Backward-compatible helper returning the numerical normalizer."""
    return build_normalizers(
        data_dir=data_dir,
        num_norm_type=num_norm_type,
    )["numerical"]


class MovingMNISTStateMultimodalDataset(Dataset):
    """A view over already-loaded multimodal arrays."""

    def __init__(
        self,
        arrays: ArrayTuple,
        image_normalizer: ArrayNormalizer,
        num_normalizer: ArrayNormalizer,
    ) -> None:
        super().__init__()

        self.x_img, self.y_img, self.x_num, self.y_num = arrays
        self.image_normalizer = image_normalizer
        self.num_normalizer = num_normalizer

    def __len__(self) -> int:
        return len(self.x_img)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x_img_raw = torch.from_numpy(self.x_img[index]).float()
        y_img_raw = torch.from_numpy(self.y_img[index]).float()
        x_num_raw = torch.from_numpy(self.x_num[index]).float()
        y_num_raw = torch.from_numpy(self.y_num[index]).float()

        return {
            "x_img": self.image_normalizer.transform(x_img_raw),
            "y_img": self.image_normalizer.transform(y_img_raw),
            "x_num": self.num_normalizer.transform(x_num_raw),
            "y_num": self.num_normalizer.transform(y_num_raw),
            "x_img_raw": x_img_raw,
            "y_img_raw": y_img_raw,
            "x_num_raw": x_num_raw,
            "y_num_raw": y_num_raw,
        }


def _slice_arrays(
    arrays: ArrayTuple,
    start: int,
    end: int,
) -> ArrayTuple:
    return tuple(array[start:end] for array in arrays)


def _limit_arrays(
    arrays: ArrayTuple,
    usage_ratio: float,
) -> ArrayTuple:
    """Keep a configurable fraction of the samples."""
    if not 0.0 < usage_ratio <= 1.0:
        raise ValueError("usage_ratio must be in the interval (0, 1].")

    if usage_ratio >= 1.0:
        return arrays

    sample_count = len(arrays[0])
    used_count = max(1, int(round(sample_count * usage_ratio)))

    return _slice_arrays(arrays, 0, used_count)


def _validate_arrays_shape(
    split: str,
    arrays: ArrayTuple,
    input_len_img: int | None,
    input_len_num: int | None,
    pred_len_img: int | None,
    pred_len_num: int | None,
    img_channels: int | None,
    num_vars: int | None,
) -> None:
    """Validate loaded arrays against the current forecasting protocol."""
    x_img, y_img, x_num, y_num = arrays
    checks = [
        ("x_img time length", x_img.shape[1], input_len_img),
        ("y_img time length", y_img.shape[1], pred_len_img),
        ("x_num time length", x_num.shape[1], input_len_num),
        ("y_num time length", y_num.shape[1], pred_len_num),
        ("image channels", x_img.shape[-1], img_channels),
        ("numerical variables", x_num.shape[-1], num_vars),
    ]

    for name, actual, expected in checks:
        if expected is not None and actual != expected:
            raise ValueError(
                f"{split} {name} mismatch: expected {expected}, got "
                f"{actual}. The current default protocol is 15 historical "
                "steps -> 15 forecast steps. Regenerate train.npz, val.npz "
                "and test.npz with scripts/generate_moving_mnist_state.py "
                "or pass matching --input_len_* / --pred_len_* values."
            )

    if img_channels is not None and y_img.shape[-1] != img_channels:
        raise ValueError(
            f"{split} y_img channels mismatch: expected {img_channels}, " f"got {y_img.shape[-1]}."
        )
    if num_vars is not None and y_num.shape[-1] != num_vars:
        raise ValueError(
            f"{split} y_num variables mismatch: expected {num_vars}, " f"got {y_num.shape[-1]}."
        )


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    img_norm_type: str = "scale_01",
    num_norm_type: str = "zscore",
    prior_fraction: float = 0.125,
    train_split_gap: int = 0,
    data_usage_ratio: float = 1.0,
    test_usage_ratio: float = 1.0,
    input_len_img: int | None = None,
    input_len_num: int | None = None,
    pred_len_img: int | None = None,
    pred_len_num: int | None = None,
    img_channels: int | None = None,
    num_vars: int | None = None,
    normalizers: Dict[str, ArrayNormalizer] | None = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, ArrayNormalizer]]:
    """Build loaders using official validation/test files.

    Only the official training file is divided, in order, between prior
    training and GeminiNet training. Arrays are loaded once and shared by
    dataset views.
    """
    arrays = load_dataset_files(data_dir)
    train_arrays = arrays["train"]

    prior_start, prior_end = split_range(
        "prior_train",
        len(train_arrays[0]),
        prior_fraction=prior_fraction,
        split_gap=train_split_gap,
    )
    gemini_start, gemini_end = split_range(
        "gemini_train",
        len(train_arrays[0]),
        prior_fraction=prior_fraction,
        split_gap=train_split_gap,
    )

    split_arrays = {
        "prior_train": _limit_arrays(
            _slice_arrays(
                train_arrays,
                prior_start,
                prior_end,
            ),
            data_usage_ratio,
        ),
        "gemini_train": _limit_arrays(
            _slice_arrays(
                train_arrays,
                gemini_start,
                gemini_end,
            ),
            data_usage_ratio,
        ),
        "val": _limit_arrays(arrays["val"], data_usage_ratio),
        "test": _limit_arrays(arrays["test"], test_usage_ratio),
    }

    for split, current_arrays in split_arrays.items():
        _validate_arrays_shape(
            split=split,
            arrays=current_arrays,
            input_len_img=input_len_img,
            input_len_num=input_len_num,
            pred_len_img=pred_len_img,
            pred_len_num=pred_len_num,
            img_channels=img_channels,
            num_vars=num_vars,
        )

    if normalizers is None:
        normalizer_train_arrays = tuple(
            np.concatenate(
                [
                    split_arrays["prior_train"][array_index],
                    split_arrays["gemini_train"][array_index],
                ],
                axis=0,
            )
            for array_index in range(len(train_arrays))
        )
        normalizers = build_normalizers(
            data_dir=data_dir,
            img_norm_type=img_norm_type,
            num_norm_type=num_norm_type,
            train_arrays=normalizer_train_arrays,
        )

    loaders: Dict[str, DataLoader] = {}

    for split, current_arrays in split_arrays.items():
        dataset = MovingMNISTStateMultimodalDataset(
            arrays=current_arrays,
            image_normalizer=normalizers["image"],
            num_normalizer=normalizers["numerical"],
        )

        loaders[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split in {"prior_train", "gemini_train"},
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return loaders, normalizers