"""Task-aware evaluation: the metric set is chosen from the model type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn import metrics

from .registry import (
    BINARY_CLASSIFICATION,
    CLUSTERING,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)


@dataclass
class EvaluationResult:
    task: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    report: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"metric": list(self.metrics), "value": list(self.metrics.values())})

    def format(self) -> str:
        lines = [f"Evaluation ({self.task})", "-" * 60]
        for name, value in self.metrics.items():
            lines.append(f"{name:<32} {value: .6f}" if isinstance(value, float) else f"{name:<32} {value}")
        for title, table in self.tables.items():
            lines += ["", title, table.to_string()]
        if self.report:
            lines += ["", "Per-class report", self.report]
        return "\n".join(lines)


def evaluate(
    task: str,
    y_true: Optional[pd.Series],
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    x_features: Optional[pd.DataFrame] = None,
    estimator: Optional[Any] = None,
) -> EvaluationResult:
    """Dispatch to the evaluation designed for the given task."""
    if task == REGRESSION:
        return _evaluate_regression(y_true, y_pred)
    if task in (BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION):
        return _evaluate_classification(task, y_true, y_pred, y_score)
    if task == CLUSTERING:
        return _evaluate_clustering(y_pred, x_features, y_true, estimator)
    raise KeyError(f"No evaluation defined for task '{task}'.")


def _evaluate_regression(y_true, y_pred) -> EvaluationResult:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(metrics.mean_squared_error(y_true, y_pred))
    scores = {
        "r2": float(metrics.r2_score(y_true, y_pred)),
        "explained_variance": float(metrics.explained_variance_score(y_true, y_pred)),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "median_absolute_error": float(metrics.median_absolute_error(y_true, y_pred)),
        "max_error": float(metrics.max_error(y_true, y_pred)),
    }
    if np.all(y_true != 0):
        scores["mape"] = float(metrics.mean_absolute_percentage_error(y_true, y_pred))

    residuals = y_true - y_pred
    table = pd.DataFrame(
        {
            "actual": y_true[:10],
            "predicted": np.round(y_pred[:10], 4),
            "residual": np.round(residuals[:10], 4),
        }
    )
    return EvaluationResult(REGRESSION, scores, {"First 10 predictions": table})


def _evaluate_classification(task, y_true, y_pred, y_score) -> EvaluationResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    average = "binary" if task == BINARY_CLASSIFICATION else "macro"
    labels = np.unique(np.concatenate([y_true, y_pred]))

    scores: Dict[str, float] = {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(metrics.cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(metrics.matthews_corrcoef(y_true, y_pred)),
    }

    kwargs = {"average": average, "zero_division": 0}
    if task == BINARY_CLASSIFICATION:
        kwargs["pos_label"] = labels[-1]
    scores[f"precision_{average}"] = float(metrics.precision_score(y_true, y_pred, **kwargs))
    scores[f"recall_{average}"] = float(metrics.recall_score(y_true, y_pred, **kwargs))
    scores[f"f1_{average}"] = float(metrics.f1_score(y_true, y_pred, **kwargs))

    if task == MULTICLASS_CLASSIFICATION:
        scores["f1_weighted"] = float(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0))

    scores.update(_probability_scores(task, y_true, y_score, labels))

    confusion = pd.DataFrame(
        metrics.confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"actual:{label}" for label in labels],
        columns=[f"pred:{label}" for label in labels],
    )
    report = metrics.classification_report(y_true, y_pred, zero_division=0)
    return EvaluationResult(task, scores, {"Confusion matrix": confusion}, report)


def _probability_scores(task, y_true, y_score, labels) -> Dict[str, float]:
    if y_score is None:
        return {}
    try:
        if task == BINARY_CLASSIFICATION:
            positive = y_score[:, 1] if y_score.ndim == 2 else y_score
            binary_true = (y_true == labels[-1]).astype(int)
            return {
                "roc_auc": float(metrics.roc_auc_score(binary_true, positive)),
                "average_precision": float(metrics.average_precision_score(binary_true, positive)),
                "log_loss": float(metrics.log_loss(binary_true, positive)),
            }
        if y_score.ndim == 2 and y_score.shape[1] == len(labels):
            return {
                "roc_auc_ovr_macro": float(
                    metrics.roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=labels)
                ),
                "log_loss": float(metrics.log_loss(y_true, y_score, labels=labels)),
            }
    except ValueError:
        # e.g. a class missing from the validation split - skip the probability metrics.
        return {}
    return {}


def _evaluate_clustering(labels_pred, x_features, y_true, estimator) -> EvaluationResult:
    labels_pred = np.asarray(labels_pred)
    unique, counts = np.unique(labels_pred, return_counts=True)
    n_clusters = int(len([u for u in unique if u != -1]))

    scores: Dict[str, float] = {
        "n_clusters": n_clusters,
        "n_noise_points": int((labels_pred == -1).sum()),
    }

    if x_features is not None and n_clusters > 1:
        mask = labels_pred != -1
        data = np.asarray(x_features)[mask]
        labels = labels_pred[mask]
        if len(np.unique(labels)) > 1:
            scores["silhouette"] = float(metrics.silhouette_score(data, labels))
            scores["calinski_harabasz"] = float(metrics.calinski_harabasz_score(data, labels))
            scores["davies_bouldin"] = float(metrics.davies_bouldin_score(data, labels))

    if estimator is not None and hasattr(estimator, "inertia_"):
        scores["inertia"] = float(estimator.inertia_)
    if estimator is not None and hasattr(estimator, "bic") and x_features is not None:
        try:
            scores["bic"] = float(estimator.bic(np.asarray(x_features)))
            scores["aic"] = float(estimator.aic(np.asarray(x_features)))
        except Exception:
            pass

    if y_true is not None:
        truth = np.asarray(y_true)
        scores["adjusted_rand_index"] = float(metrics.adjusted_rand_score(truth, labels_pred))
        scores["normalized_mutual_info"] = float(metrics.normalized_mutual_info_score(truth, labels_pred))
        scores["homogeneity"] = float(metrics.homogeneity_score(truth, labels_pred))
        scores["completeness"] = float(metrics.completeness_score(truth, labels_pred))
        scores["v_measure"] = float(metrics.v_measure_score(truth, labels_pred))

    sizes = pd.DataFrame({"cluster": unique, "size": counts})
    return EvaluationResult(CLUSTERING, scores, {"Cluster sizes": sizes})
