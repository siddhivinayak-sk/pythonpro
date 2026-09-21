"""Training, persistence and prediction for the testbed."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from .config import ExperimentConfig
from .dataset import load_dataset, prepare_split, split_feature_types
from .evaluation import EvaluationResult, evaluate
from .preprocessing import build_preprocessor
from .registry import CLUSTERING, FIT_ONLY_CLUSTERERS, REGRESSION, get_algorithm

MODEL_SUFFIX = ".joblib"


@dataclass
class TrainedModel:
    """A fitted pipeline plus everything needed to reuse and explain it."""

    pipeline: Pipeline
    config: ExperimentConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation: Optional[EvaluationResult] = None

    @property
    def task(self) -> str:
        return self.config.task

    @property
    def feature_columns(self) -> List[str]:
        return list(self.config.feature_columns)

    def summary(self) -> str:
        spec = get_algorithm(self.config.algorithm)
        lines = [
            "Model summary",
            "-" * 60,
            f"algorithm      : {spec.label} ({spec.key})",
            f"task           : {self.config.task}",
            f"category       : {spec.category}",
            f"dataset        : {self.config.csv_path}",
            f"label column   : {self.config.label_column or '(none - unsupervised)'}",
            f"features       : {', '.join(self.config.feature_columns)}",
            f"training rows  : {self.metadata.get('n_train_rows')}",
            f"validation rows: {self.metadata.get('n_test_rows')}",
            f"trained at     : {self.metadata.get('trained_at')}",
        ]
        if self.config.is_image:
            lines.insert(
                4,
                f"input mode     : image ({self.config.image_featurizer}, "
                f"{self.config.image_size[0]}x{self.config.image_size[1]}, {self.config.image_color})",
            )
        if spec.notes:
            lines.append(f"note           : {spec.notes}")
        return "\n".join(lines)


def build_pipeline(frame: pd.DataFrame, config: ExperimentConfig) -> Pipeline:
    spec = get_algorithm(config.algorithm)
    if config.task not in spec.tasks:
        raise ValueError(f"Algorithm '{spec.key}' does not support task '{config.task}'.")
    preprocessor = _build_image_preprocessor(config, spec) if config.is_image else _build_tabular_preprocessor(frame, config, spec)
    estimator = spec.build(config.hyperparameters)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _build_tabular_preprocessor(frame: pd.DataFrame, config: ExperimentConfig, spec):
    numeric, categorical = split_feature_types(frame, list(config.feature_columns))
    return build_preprocessor(numeric, categorical, scaler=spec.scaler)


def _build_image_preprocessor(config: ExperimentConfig, spec) -> Pipeline:
    from sklearn.preprocessing import StandardScaler

    from .images import ImageFeaturizer

    steps = [
        (
            "featurizer",
            ImageFeaturizer(
                image_size=config.image_size,
                color_mode=config.image_color,
                method=config.image_featurizer,
            ),
        )
    ]
    # Pixel and HOG features are already in [0, 1]; only standardise when the model wants it.
    if spec.scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def run_experiment(config: ExperimentConfig, cross_validation_folds: int = 0) -> TrainedModel:
    """Load the data, split it, fit the pipeline and evaluate it for the chosen task."""
    frame = load_dataset(config)
    split = prepare_split(
        frame,
        list(config.feature_columns),
        config.label_column,
        config.task,
        config.test_size,
        config.random_state,
        config.stratify,
        config.dropna_target,
    )
    pipeline = build_pipeline(frame, config)

    metadata: Dict[str, Any] = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "n_train_rows": int(len(split.x_train)),
        "n_test_rows": int(len(split.x_test)) if config.task != CLUSTERING else 0,
    }

    if config.task == CLUSTERING:
        evaluation = _fit_and_score_clustering(pipeline, split, config)
    else:
        pipeline.fit(split.x_train, split.y_train)
        y_pred = pipeline.predict(split.x_test)
        y_score = _decision_scores(pipeline, split.x_test)
        evaluation = evaluate(config.task, split.y_test, y_pred, y_score)
        metadata["classes"] = _classes(pipeline)
        if cross_validation_folds and cross_validation_folds > 1:
            scoring = "r2" if config.task == REGRESSION else "accuracy"
            cv_scores = cross_val_score(
                build_pipeline(frame, config),
                split.x_train,
                split.y_train,
                cv=cross_validation_folds,
                scoring=scoring,
            )
            evaluation.metrics[f"cv_{scoring}_mean"] = float(np.mean(cv_scores))
            evaluation.metrics[f"cv_{scoring}_std"] = float(np.std(cv_scores))

    return TrainedModel(pipeline=pipeline, config=config, metadata=metadata, evaluation=evaluation)


def _fit_and_score_clustering(pipeline: Pipeline, split, config: ExperimentConfig) -> EvaluationResult:
    model = pipeline.named_steps["model"]
    transformed = pipeline.named_steps["preprocessor"].fit_transform(split.x_train)
    if hasattr(model, "fit_predict"):
        labels = model.fit_predict(transformed)
    else:  # e.g. GaussianMixture
        model.fit(transformed)
        labels = model.predict(transformed)
    return evaluate(
        CLUSTERING,
        y_true=split.y_train,
        y_pred=labels,
        x_features=pd.DataFrame(transformed),
        estimator=model,
    )


def _classes(pipeline: Pipeline) -> Optional[List[Any]]:
    classes = getattr(pipeline.named_steps["model"], "classes_", None)
    return None if classes is None else [c.item() if hasattr(c, "item") else c for c in classes]


def _decision_scores(pipeline: Pipeline, x: pd.DataFrame) -> Optional[np.ndarray]:
    """Probabilities when available, otherwise decision function values."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(pipeline.predict_proba(x))
        except Exception:
            return None
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(pipeline.decision_function(x))
        except Exception:
            return None
    return None


def save_model(model: TrainedModel, path: str) -> Path:
    """Persist the fitted pipeline plus a readable JSON sidecar."""
    target = Path(path).expanduser()
    if target.suffix != MODEL_SUFFIX:
        target = target.with_suffix(MODEL_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pipeline": model.pipeline,
        "config": model.config.to_dict(),
        "metadata": model.metadata,
        "metrics": dict(model.evaluation.metrics) if model.evaluation else {},
    }
    joblib.dump(payload, target)

    sidecar = target.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {"config": payload["config"], "metadata": payload["metadata"], "metrics": payload["metrics"]},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return target


def load_model(path: str) -> TrainedModel:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Model file not found: {source}")
    payload = joblib.load(source)
    config = ExperimentConfig.from_dict(payload["config"])
    evaluation = EvaluationResult(task=config.task, metrics=payload.get("metrics", {}))
    return TrainedModel(payload["pipeline"], config, payload.get("metadata", {}), evaluation)


def predict_frame(model: TrainedModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Score new rows; returns the input plus a prediction (and probability) column."""
    missing = [c for c in model.feature_columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Input data is missing required feature columns: {missing}")

    features = frame[model.feature_columns]
    pipeline = model.pipeline

    if model.task == CLUSTERING:
        if model.config.algorithm in FIT_ONLY_CLUSTERERS:
            raise ValueError(
                f"'{model.config.algorithm}' cannot assign clusters to unseen rows. "
                "Use k-means, BIRCH, mini-batch k-means or a Gaussian mixture for prediction."
            )
        transformed = pipeline.named_steps["preprocessor"].transform(features)
        predictions = pipeline.named_steps["model"].predict(transformed)
        result = frame.copy()
        result["cluster"] = predictions
        return result

    result = frame.copy()
    result["prediction"] = pipeline.predict(features)
    model_step = pipeline.named_steps["model"]
    if hasattr(model_step, "predict_proba"):
        proba = pipeline.predict_proba(features)
        for index, class_label in enumerate(model_step.classes_):
            result[f"proba_{class_label}"] = proba[:, index]
    return result


def predict_csv(model_path: str, csv_path: str, output_path: Optional[str] = None, separator: str = ",") -> pd.DataFrame:
    """Load a saved model and score every row of a CSV (or every image of a folder/manifest)."""
    model = load_model(model_path)
    frame = load_dataset(_source_config(model.config, csv_path, separator))
    scored = predict_frame(model, frame)
    if output_path:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out, index=False)
    return scored


def _source_config(config: ExperimentConfig, source: str, separator: str) -> ExperimentConfig:
    replacement = ExperimentConfig.from_dict(config.to_dict())
    replacement.csv_path = source
    replacement.csv_separator = separator
    return replacement


def predict_images(model: TrainedModel, paths: Sequence[str]) -> pd.DataFrame:
    """Score individual image files with a model trained in image mode."""
    if not model.config.is_image:
        raise ValueError("This model was not trained on images.")
    from .images import IMAGE_PATH_COLUMN

    return predict_frame(model, pd.DataFrame({IMAGE_PATH_COLUMN: [str(p) for p in paths]}))
