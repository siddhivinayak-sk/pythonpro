"""Configuration objects describing a single testbed experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

TABULAR = "tabular"
IMAGE = "image"
INPUT_MODES = (TABULAR, IMAGE)


@dataclass
class ExperimentConfig:
    """Everything the testbed needs to reproduce one training run."""

    csv_path: str  # tabular: the CSV; image: a folder-per-class directory or a manifest CSV
    algorithm: str
    task: str
    feature_columns: List[str]
    label_column: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    csv_separator: str = ","
    dropna_target: bool = True
    # image mode only
    input_mode: str = TABULAR
    image_size: Tuple[int, int] = (32, 32)
    image_color: str = "gray"  # "gray" | "rgb"
    image_featurizer: str = "flatten"  # "flatten" | "hog" | "stats"

    def __post_init__(self) -> None:
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got '{self.input_mode}'.")
        self.image_size = tuple(self.image_size)
        if self.image_color not in ("gray", "rgb"):
            raise ValueError("image_color must be 'gray' or 'rgb'.")
        if not self.feature_columns:
            raise ValueError("At least one feature column is required.")
        if not 0.0 < self.test_size < 0.9:
            raise ValueError("test_size must be between 0 and 0.9 (exclusive).")
        if self.task != "clustering" and not self.label_column:
            raise ValueError(f"Task '{self.task}' requires a label column.")
        if self.label_column and self.label_column in self.feature_columns:
            raise ValueError("The label column cannot also be used as a feature.")

    @property
    def is_image(self) -> bool:
        return self.input_mode == IMAGE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "algorithm": self.algorithm,
            "task": self.task,
            "feature_columns": list(self.feature_columns),
            "label_column": self.label_column,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify": self.stratify,
            "hyperparameters": dict(self.hyperparameters),
            "csv_separator": self.csv_separator,
            "dropna_target": self.dropna_target,
            "input_mode": self.input_mode,
            "image_size": list(self.image_size),
            "image_color": self.image_color,
            "image_featurizer": self.image_featurizer,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def image_config(
    source: str,
    algorithm: str,
    task: str,
    label_column: Optional[str] = "label",
    **kwargs: Any,
) -> ExperimentConfig:
    """Convenience builder: image experiments always use the 'image_path' feature column."""
    from .images import IMAGE_PATH_COLUMN

    return ExperimentConfig(
        csv_path=source,
        algorithm=algorithm,
        task=task,
        feature_columns=[IMAGE_PATH_COLUMN],
        label_column=label_column,
        input_mode=IMAGE,
        **kwargs,
    )
