from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONLogger:
    """Save epoch-wise training metrics as a JSON file.

    The saved JSON format is:

        {
            "epoch": [1, 2, 3],
            "train_img_mse": [0.12, 0.08, 0.05],
            "val_img_mse": [0.14, 0.10, 0.07]
        }

    Each metric is stored as an independent list, so later visualization is
    usually more stable than CSV-based reading.
    """

    def __init__(
        self,
        path: str | Path,
        reset: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists() and not reset:
            with open(self.path, "r", encoding="utf-8") as file:
                self.history = json.load(file)
        else:
            self.history = {}

    def log(self, row: dict[str, Any]) -> None:
        """Append one row of metrics and save immediately."""
        for key, value in row.items():
            if key not in self.history:
                self.history[key] = []

            self.history[key].append(self._to_json_serializable(value))

        self.save()

    def save(self) -> None:
        """Write metric history to disk."""
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(
                self.history,
                file,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def _to_json_serializable(value: Any) -> Any:
        """Convert common tensor / numpy scalar values to Python values."""
        if hasattr(value, "item"):
            return value.item()

        return value


def save_json(
    path: str | Path,
    data: dict,
) -> None:
    """Save a dictionary as a JSON file.

    Used for final test metrics, configuration summaries, and other one-time
    results.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_data = {key: JSONLogger._to_json_serializable(value) for key, value in data.items()}

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            clean_data,
            file,
            indent=2,
            ensure_ascii=False,
        )