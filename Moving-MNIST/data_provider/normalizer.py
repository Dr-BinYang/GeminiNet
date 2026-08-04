from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class ArrayNormalizer:
    """Feature-wise normalizer for image fields and numerical sequences.

    Statistics are computed over every axis except the final feature/channel
    axis. The same class can therefore normalize arrays shaped [..., C] or
    [..., D], including the project's [N, T, H, W, C] image layout.
    """

    SUPPORTED_TYPES = {
        "none",
        "scale_01",
        "minmax",
        "zscore",
        "robust",
        "log1p_zscore",
    }

    def __init__(
        self,
        norm_type: str = "zscore",
        eps: float = 1e-6,
        scale_01_divisor: float = 255.0,
        robust_max_samples: int = 2_000_000,
    ) -> None:
        norm_type = norm_type.lower()

        if norm_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported normalization type: {norm_type}. "
                f"Choose from {sorted(self.SUPPORTED_TYPES)}."
            )

        self.norm_type = norm_type
        self.eps = float(eps)
        self.scale_01_divisor = float(scale_01_divisor)
        self.robust_max_samples = int(robust_max_samples)
        self.stats: dict[str, np.ndarray] = {}
        self.is_fitted = norm_type in {"none", "scale_01"}

    @staticmethod
    def _as_arrays(
        data: np.ndarray | Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        if isinstance(data, np.ndarray):
            arrays = [data]
        else:
            arrays = [np.asarray(item) for item in data]

        if not arrays:
            raise ValueError("Cannot fit a normalizer on empty data.")

        return arrays

    @staticmethod
    def _flatten_features(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)

        if array.ndim == 0:
            return array.reshape(1, 1)

        if array.ndim == 1:
            return array.reshape(-1, 1)

        return array.reshape(-1, array.shape[-1])

    @staticmethod
    def _check_non_negative(array: np.ndarray) -> None:
        if np.any(array < 0):
            raise ValueError(
                "log1p_zscore requires non-negative data, but negative " "values were found."
            )

    def fit(
        self,
        data: np.ndarray | Sequence[np.ndarray],
    ) -> "ArrayNormalizer":
        """Fit statistics from training-range arrays only."""
        arrays = self._as_arrays(data)

        if self.norm_type in {"none", "scale_01"}:
            self.stats = {}
            self.is_fitted = True
            return self

        flattened = [self._flatten_features(array) for array in arrays]

        if self.norm_type == "minmax":
            self.stats = {
                "min": np.minimum.reduce([array.min(axis=0) for array in flattened]).astype(
                    np.float32
                ),
                "max": np.maximum.reduce([array.max(axis=0) for array in flattened]).astype(
                    np.float32
                ),
            }

        elif self.norm_type in {"zscore", "log1p_zscore"}:
            feature_count = flattened[0].shape[-1]
            count = 0
            total = np.zeros(feature_count, dtype=np.float64)
            squared_total = np.zeros(feature_count, dtype=np.float64)
            chunk_size = 1_000_000

            for array in flattened:
                if self.norm_type == "log1p_zscore":
                    self._check_non_negative(array)

                for start in range(0, array.shape[0], chunk_size):
                    chunk = array[start : start + chunk_size].astype(
                        np.float64,
                        copy=False,
                    )

                    if self.norm_type == "log1p_zscore":
                        chunk = np.log1p(chunk)

                    count += chunk.shape[0]
                    total += chunk.sum(axis=0)
                    squared_total += np.square(chunk).sum(axis=0)

            mean = total / max(1, count)
            variance = squared_total / max(1, count) - np.square(mean)

            self.stats = {
                "mean": mean.astype(np.float32),
                "std": np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
            }

        elif self.norm_type == "robust":
            samples_per_array = max(
                1,
                self.robust_max_samples // len(flattened),
            )
            sampled = []

            for array in flattened:
                if array.shape[0] > samples_per_array:
                    indices = np.linspace(
                        0,
                        array.shape[0] - 1,
                        num=samples_per_array,
                        dtype=np.int64,
                    )
                    array = array[indices]
                sampled.append(array)

            combined = np.concatenate(sampled, axis=0)
            q1 = np.percentile(combined, 25.0, axis=0)
            median = np.median(combined, axis=0)
            q3 = np.percentile(combined, 75.0, axis=0)

            self.stats = {
                "median": median.astype(np.float32),
                "iqr": (q3 - q1).astype(np.float32),
            }

        self.is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before use.")

    def _stat_like(
        self,
        name: str,
        data: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        value = self.stats[name]

        if torch.is_tensor(data):
            return torch.as_tensor(
                value,
                dtype=data.dtype if data.is_floating_point() else torch.float32,
                device=data.device,
            )

        return value

    def transform(
        self,
        data: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Normalize an array or tensor without changing its shape."""
        self._require_fitted()
        data = (
            data.float()
            if torch.is_tensor(data)
            else np.asarray(
                data,
                dtype=np.float32,
            )
        )

        if self.norm_type == "none":
            return data

        if self.norm_type == "scale_01":
            return data / (self.scale_01_divisor + self.eps)

        if self.norm_type == "minmax":
            minimum = self._stat_like("min", data)
            maximum = self._stat_like("max", data)
            return (data - minimum) / (maximum - minimum + self.eps)

        if self.norm_type == "zscore":
            mean = self._stat_like("mean", data)
            std = self._stat_like("std", data)
            return (data - mean) / (std + self.eps)

        if self.norm_type == "robust":
            median = self._stat_like("median", data)
            iqr = self._stat_like("iqr", data)
            return (data - median) / (iqr + self.eps)

        if torch.is_tensor(data):
            if torch.any(data < 0):
                raise ValueError(
                    "log1p_zscore requires non-negative data, but negative " "values were found."
                )
            transformed = torch.log1p(data)
        else:
            self._check_non_negative(data)
            transformed = np.log1p(data)

        mean = self._stat_like("mean", data)
        std = self._stat_like("std", data)
        return (transformed - mean) / (std + self.eps)

    def inverse_transform(
        self,
        data: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Restore normalized data to its original scale."""
        self._require_fitted()
        data = (
            data.float()
            if torch.is_tensor(data)
            else np.asarray(
                data,
                dtype=np.float32,
            )
        )

        if self.norm_type == "none":
            return data

        if self.norm_type == "scale_01":
            return data * (self.scale_01_divisor + self.eps)

        if self.norm_type == "minmax":
            minimum = self._stat_like("min", data)
            maximum = self._stat_like("max", data)
            return data * (maximum - minimum + self.eps) + minimum

        if self.norm_type == "zscore":
            mean = self._stat_like("mean", data)
            std = self._stat_like("std", data)
            return data * (std + self.eps) + mean

        if self.norm_type == "robust":
            median = self._stat_like("median", data)
            iqr = self._stat_like("iqr", data)
            return data * (iqr + self.eps) + median

        mean = self._stat_like("mean", data)
        std = self._stat_like("std", data)
        transformed = data * (std + self.eps) + mean

        if torch.is_tensor(transformed):
            return torch.expm1(transformed)

        return np.expm1(transformed)

    # Backward-compatible names used by the existing dataset code.
    def normalize(
        self,
        data: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        return self.transform(data)

    def denormalize(
        self,
        data: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        return self.inverse_transform(data)

    def to_dict(self) -> dict:
        return {
            "norm_type": self.norm_type,
            "eps": self.eps,
            "scale_01_divisor": self.scale_01_divisor,
            "robust_max_samples": self.robust_max_samples,
            "stats": {key: value.tolist() for key, value in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArrayNormalizer":
        normalizer = cls(
            norm_type=data["norm_type"],
            eps=data.get("eps", 1e-6),
            scale_01_divisor=data.get("scale_01_divisor", 255.0),
            robust_max_samples=data.get(
                "robust_max_samples",
                2_000_000,
            ),
        )
        normalizer.stats = {
            key: np.asarray(value, dtype=np.float32) for key, value in data.get("stats", {}).items()
        }
        normalizer.is_fitted = True
        return normalizer


class NumericalNormalizer(ArrayNormalizer):
    """Backward-compatible z-score numerical normalizer."""

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(norm_type="zscore", eps=eps)
        self.stats = {
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
        }
        self.mean_np = self.stats["mean"]
        self.std_np = self.stats["std"]
        self.is_fitted = True

    @classmethod
    def from_arrays(
        cls,
        x_num: np.ndarray,
        y_num: np.ndarray,
    ) -> "NumericalNormalizer":
        fitted = ArrayNormalizer(norm_type="zscore").fit([x_num, y_num])
        return cls(
            mean=fitted.stats["mean"],
            std=fitted.stats["std"],
            eps=fitted.eps,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "NumericalNormalizer":
        if "stats" in data:
            stats = data["stats"]
            return cls(
                mean=np.asarray(stats["mean"], dtype=np.float32),
                std=np.asarray(stats["std"], dtype=np.float32),
                eps=data.get("eps", 1e-6),
            )

        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            eps=data.get("eps", 1e-6),
        )

    def to_dict(self) -> dict:
        """Preserve the legacy numerical-normalizer serialization format."""
        return {
            "mean": self.mean_np.tolist(),
            "std": self.std_np.tolist(),
            "eps": self.eps,
        }