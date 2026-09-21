"""Builds the preprocessing pipeline (imputation, scaling, one-hot encoding)."""

from __future__ import annotations

from typing import List

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def _one_hot_encoder() -> OneHotEncoder:
    """OneHotEncoder with a dense output across scikit-learn versions."""
    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    scaler: str = "standard",
) -> ColumnTransformer:
    """Numeric -> median impute + scale, categorical -> mode impute + one-hot."""
    steps = []

    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scaler == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
        steps.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ]
        if scaler == "minmax":
            categorical_steps.append(("scaler", MinMaxScaler()))
        steps.append(("categorical", Pipeline(categorical_steps), categorical_columns))

    if not steps:
        raise ValueError("No feature columns were provided to the preprocessor.")

    return ColumnTransformer(transformers=steps, remainder="drop")
