"""CSV loading, column profiling, task inference and train/validation splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .registry import (
    BINARY_CLASSIFICATION,
    CLUSTERING,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_unique: int
    n_missing: int
    is_numeric: bool
    sample: str

    def describe(self) -> str:
        return (
            f"{self.name:<28} {self.dtype:<10} unique={self.n_unique:<6} "
            f"missing={self.n_missing:<5} e.g. {self.sample}"
        )


@dataclass
class SplitData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]


def load_csv(path: str, separator: str = ",") -> pd.DataFrame:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    frame = pd.read_csv(csv_path, sep=separator)
    if frame.empty:
        raise ValueError(f"CSV file '{csv_path}' contains no rows.")
    return frame


def load_dataset(config) -> pd.DataFrame:
    """Load the experiment's data source according to its input mode."""
    if getattr(config, "input_mode", "tabular") == "image":
        from .images import load_images

        return load_images(config.csv_path, config.csv_separator)
    return load_csv(config.csv_path, config.csv_separator)


def profile_columns(frame: pd.DataFrame) -> List[ColumnProfile]:
    profiles = []
    for name in frame.columns:
        series = frame[name]
        sample = series.dropna()
        profiles.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                n_unique=int(series.nunique(dropna=True)),
                n_missing=int(series.isna().sum()),
                is_numeric=bool(pd.api.types.is_numeric_dtype(series)),
                sample="" if sample.empty else str(sample.iloc[0]),
            )
        )
    return profiles


def split_feature_types(frame: pd.DataFrame, feature_columns: List[str]) -> Tuple[List[str], List[str]]:
    """Return (numeric_columns, categorical_columns) for the chosen features."""
    numeric, categorical = [], []
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def infer_task(frame: pd.DataFrame, label_column: Optional[str]) -> str:
    """Guess the learning task from the label column (clustering when no label)."""
    if not label_column:
        return CLUSTERING
    series = frame[label_column].dropna()
    n_unique = series.nunique()
    if n_unique < 2:
        raise ValueError(f"Label column '{label_column}' has fewer than 2 distinct values.")
    if not pd.api.types.is_numeric_dtype(series):
        return BINARY_CLASSIFICATION if n_unique == 2 else MULTICLASS_CLASSIFICATION
    is_discrete = pd.api.types.is_integer_dtype(series) or np.allclose(series, series.round())
    if is_discrete and n_unique <= max(20, int(0.05 * len(series))):
        return BINARY_CLASSIFICATION if n_unique == 2 else MULTICLASS_CLASSIFICATION
    return REGRESSION


def prepare_split(
    frame: pd.DataFrame,
    feature_columns: List[str],
    label_column: Optional[str],
    task: str,
    test_size: float,
    random_state: int,
    stratify: bool = True,
    dropna_target: bool = True,
) -> SplitData:
    missing = [c for c in feature_columns + ([label_column] if label_column else []) if c not in frame.columns]
    if missing:
        raise KeyError(f"Columns missing from the CSV: {missing}")

    working = frame.copy()
    if label_column and dropna_target:
        working = working.dropna(subset=[label_column])
        if working.empty:
            raise ValueError(f"No rows left after dropping missing values of '{label_column}'.")

    features = working[feature_columns]

    if task == CLUSTERING:
        # Unsupervised: everything is used for fitting; the label (if any) is kept for scoring only.
        labels = working[label_column] if label_column else None
        return SplitData(features, features, labels, labels)

    target = working[label_column]
    stratify_on = target if (stratify and task != REGRESSION and target.value_counts().min() >= 2) else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_on,
    )
    return SplitData(x_train, x_test, y_train, y_test)
