from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

SUPPORTED_NORM_TYPES = [
    "none",
    "scale_01",
    "minmax",
    "zscore",
    "robust",
    "log1p_zscore",
]


@dataclass
class ArrayNormalizer:
    """General array normalizer for image-like or numerical sequence data.

    The normalizer is fitted only on the training range and then reused for
    validation/test sets. It supports both ordinary image pixels and physical
    field-like image data.

    Supported normalization types:
        none:
            Do not normalize.
        scale_01:
            Divide by value_scale, usually 255 for uint8 images.
        minmax:
            (x - min) / (max - min), fitted from training range.
        zscore:
            (x - mean) / std, fitted from training range.
        robust:
            (x - median) / IQR, fitted from training range.
        log1p_zscore:
            log1p(x) followed by z-score. Use for non-negative long-tail data.
    """

    norm_type: str = "zscore"
    mean: np.ndarray | float | None = None
    std: np.ndarray | float | None = None
    min_value: np.ndarray | float | None = None
    max_value: np.ndarray | float | None = None
    median: np.ndarray | float | None = None
    iqr: np.ndarray | float | None = None
    value_scale: float = 255.0
    eps: float = 1e-6
    reduce_axes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.norm_type = self.norm_type.lower()

        if self.norm_type not in SUPPORTED_NORM_TYPES:
            raise ValueError(
                f"Unsupported norm_type={self.norm_type}. " f"Supported: {SUPPORTED_NORM_TYPES}"
            )

    @staticmethod
    def _as_np(value):
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)

    @classmethod
    def fit(
        cls,
        arrays: list[np.ndarray],
        norm_type: str = "zscore",
        value_scale: float = 255.0,
        eps: float = 1e-6,
        reduce_axes: tuple[int, ...] | None = None,
    ) -> "ArrayNormalizer":
        """Fit normalizer from one or more arrays.

        Args:
            arrays:
                Arrays from the training range only.
            norm_type:
                Normalization mode.
            value_scale:
                Used by scale_01.
            reduce_axes:
                Axes used to compute statistics. If None, all elements are
                reduced to scalar statistics. For numerical variables, use
                all axes except the last one to obtain variable-wise stats.
        """
        norm_type = norm_type.lower()

        if norm_type not in SUPPORTED_NORM_TYPES:
            raise ValueError(
                f"Unsupported norm_type={norm_type}. " f"Supported: {SUPPORTED_NORM_TYPES}"
            )

        if len(arrays) == 0:
            raise ValueError("ArrayNormalizer.fit requires at least one array.")

        if norm_type in ["none", "scale_01"]:
            return cls(
                norm_type=norm_type,
                value_scale=value_scale,
                eps=eps,
                reduce_axes=reduce_axes,
            )

        arrays = [np.asarray(array, dtype=np.float32) for array in arrays]

        data = np.concatenate(arrays, axis=0)

        if norm_type == "log1p_zscore":
            if np.nanmin(data) < -eps:
                raise ValueError(
                    "log1p_zscore requires non-negative data. "
                    "Use zscore or robust for variables with negative values."
                )
            data_for_stats = np.log1p(np.clip(data, a_min=0.0, a_max=None))
        else:
            data_for_stats = data

        if reduce_axes is None:
            reduce_axes = tuple(range(data_for_stats.ndim))

        if norm_type == "minmax":
            min_value = data_for_stats.min(axis=reduce_axes, keepdims=False)
            max_value = data_for_stats.max(axis=reduce_axes, keepdims=False)
            return cls(
                norm_type=norm_type,
                min_value=min_value.astype(np.float32),
                max_value=max_value.astype(np.float32),
                value_scale=value_scale,
                eps=eps,
                reduce_axes=reduce_axes,
            )

        if norm_type in ["zscore", "log1p_zscore"]:
            mean = data_for_stats.mean(axis=reduce_axes, keepdims=False)
            std = data_for_stats.std(axis=reduce_axes, keepdims=False)
            std = np.maximum(std, eps)
            return cls(
                norm_type=norm_type,
                mean=mean.astype(np.float32),
                std=std.astype(np.float32),
                value_scale=value_scale,
                eps=eps,
                reduce_axes=reduce_axes,
            )

        if norm_type == "robust":
            q1 = np.percentile(data_for_stats, 25, axis=reduce_axes, keepdims=False)
            q3 = np.percentile(data_for_stats, 75, axis=reduce_axes, keepdims=False)
            median = np.median(data_for_stats, axis=reduce_axes, keepdims=False)
            iqr = np.maximum(q3 - q1, eps)
            return cls(
                norm_type=norm_type,
                median=median.astype(np.float32),
                iqr=iqr.astype(np.float32),
                value_scale=value_scale,
                eps=eps,
                reduce_axes=reduce_axes,
            )

        raise ValueError(f"Unsupported norm_type={norm_type}")

    def to_dict(self) -> dict:
        """Serialize normalization statistics."""

        def convert(value):
            if value is None:
                return None
            array = np.asarray(value)
            return array.tolist()

        return {
            "norm_type": self.norm_type,
            "mean": convert(self.mean),
            "std": convert(self.std),
            "min_value": convert(self.min_value),
            "max_value": convert(self.max_value),
            "median": convert(self.median),
            "iqr": convert(self.iqr),
            "value_scale": self.value_scale,
            "eps": self.eps,
            "reduce_axes": list(self.reduce_axes) if self.reduce_axes is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArrayNormalizer":
        """Restore normalizer from a dictionary."""
        reduce_axes = data.get("reduce_axes", None)
        if reduce_axes is not None:
            reduce_axes = tuple(reduce_axes)

        return cls(
            norm_type=data.get("norm_type", "zscore"),
            mean=cls._as_np(data.get("mean", None)),
            std=cls._as_np(data.get("std", None)),
            min_value=cls._as_np(data.get("min_value", None)),
            max_value=cls._as_np(data.get("max_value", None)),
            median=cls._as_np(data.get("median", None)),
            iqr=cls._as_np(data.get("iqr", None)),
            value_scale=float(data.get("value_scale", 255.0)),
            eps=float(data.get("eps", 1e-6)),
            reduce_axes=reduce_axes,
        )

    def _to_tensor(self, value, x: torch.Tensor) -> torch.Tensor:
        if value is None:
            raise ValueError(f"Missing statistics for norm_type={self.norm_type}")

        return torch.as_tensor(
            value,
            dtype=torch.float32,
            device=x.device,
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor while preserving its original shape."""
        x = x.float()

        if self.norm_type == "none":
            return x

        if self.norm_type == "scale_01":
            return x / max(float(self.value_scale), self.eps)

        if self.norm_type == "minmax":
            min_value = self._to_tensor(self.min_value, x)
            max_value = self._to_tensor(self.max_value, x)
            return (x - min_value) / torch.clamp(max_value - min_value, min=self.eps)

        if self.norm_type == "zscore":
            mean = self._to_tensor(self.mean, x)
            std = self._to_tensor(self.std, x)
            return (x - mean) / torch.clamp(std, min=self.eps)

        if self.norm_type == "robust":
            median = self._to_tensor(self.median, x)
            iqr = self._to_tensor(self.iqr, x)
            return (x - median) / torch.clamp(iqr, min=self.eps)

        if self.norm_type == "log1p_zscore":
            mean = self._to_tensor(self.mean, x)
            std = self._to_tensor(self.std, x)
            x_log = torch.log1p(torch.clamp(x, min=0.0))
            return (x_log - mean) / torch.clamp(std, min=self.eps)

        raise ValueError(f"Unsupported norm_type={self.norm_type}")

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Recover original scale when the transform is invertible."""
        x = x.float()

        if self.norm_type == "none":
            return x

        if self.norm_type == "scale_01":
            return x * max(float(self.value_scale), self.eps)

        if self.norm_type == "minmax":
            min_value = self._to_tensor(self.min_value, x)
            max_value = self._to_tensor(self.max_value, x)
            return x * torch.clamp(max_value - min_value, min=self.eps) + min_value

        if self.norm_type == "zscore":
            mean = self._to_tensor(self.mean, x)
            std = self._to_tensor(self.std, x)
            return x * torch.clamp(std, min=self.eps) + mean

        if self.norm_type == "robust":
            median = self._to_tensor(self.median, x)
            iqr = self._to_tensor(self.iqr, x)
            return x * torch.clamp(iqr, min=self.eps) + median

        if self.norm_type == "log1p_zscore":
            mean = self._to_tensor(self.mean, x)
            std = self._to_tensor(self.std, x)
            return torch.expm1(x * torch.clamp(std, min=self.eps) + mean)

        raise ValueError(f"Unsupported norm_type={self.norm_type}")

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Alias of denormalize for visualization/evaluation call sites."""
        return self.denormalize(x)


class NumericalNormalizer(ArrayNormalizer):
    """Backward-compatible numerical normalizer.

    By default, this is variable-wise z-score normalization. It is kept as a
    subclass so existing imports remain valid.
    """

    @classmethod
    def from_arrays(
        cls,
        x_num: np.ndarray,
        y_num: np.ndarray,
        norm_type: str = "zscore",
        eps: float = 1e-6,
    ) -> "NumericalNormalizer":
        return cls.fit_from_arrays(
            x_num=x_num,
            y_num=y_num,
            norm_type=norm_type,
            eps=eps,
        )

    @classmethod
    def fit_from_arrays(
        cls,
        x_num: np.ndarray,
        y_num: np.ndarray,
        norm_type: str = "zscore",
        eps: float = 1e-6,
    ) -> "NumericalNormalizer":
        normalizer = ArrayNormalizer.fit(
            arrays=[x_num, y_num],
            norm_type=norm_type,
            eps=eps,
            reduce_axes=(0, 1),
        )

        return cls(
            norm_type=normalizer.norm_type,
            mean=normalizer.mean,
            std=normalizer.std,
            min_value=normalizer.min_value,
            max_value=normalizer.max_value,
            median=normalizer.median,
            iqr=normalizer.iqr,
            value_scale=normalizer.value_scale,
            eps=normalizer.eps,
            reduce_axes=normalizer.reduce_axes,
        )


@dataclass
class MultimodalNormalizer:
    """Container holding image and numerical normalizers."""

    image: ArrayNormalizer
    numerical: NumericalNormalizer

    def to_dict(self) -> dict:
        return {
            "image": self.image.to_dict(),
            "numerical": self.numerical.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultimodalNormalizer":
        return cls(
            image=ArrayNormalizer.from_dict(data["image"]),
            numerical=NumericalNormalizer.from_dict(data["numerical"]),
        )