"""Optional matplotlib visualisations that match the task being demonstrated."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .registry import BINARY_CLASSIFICATION, CLASSIFICATION_TASKS, CLUSTERING, REGRESSION


def _finish(fig, output_dir: Optional[str], filename: str, show: bool) -> Optional[str]:
    saved = None
    if output_dir:
        target = Path(output_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        path = target / filename
        fig.savefig(path, dpi=120, bbox_inches="tight")
        saved = str(path)
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return saved


def plot_results(
    model,
    x: pd.DataFrame,
    y_true: Optional[pd.Series] = None,
    output_dir: Optional[str] = None,
    show: bool = False,
) -> List[str]:
    """Render the charts that make sense for the model's task."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = model.task
    saved: List[str] = []

    if model.config.is_image:
        from .images import plot_image_predictions

        frame = x.copy()
        if y_true is not None and model.config.label_column:
            frame[model.config.label_column] = y_true
        saved.extend(plot_image_predictions(model, frame, output_dir=output_dir, show=show))

    if task == REGRESSION and y_true is not None:
        y_pred = model.pipeline.predict(x)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
        limits = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
        axes[0].plot(limits, limits, "r--", linewidth=1)
        axes[0].set(xlabel="actual", ylabel="predicted", title="Predicted vs actual")
        residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolor="none")
        axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
        axes[1].set(xlabel="predicted", ylabel="residual", title="Residuals")
        fig.tight_layout()
        saved.append(_finish(fig, output_dir, "regression_diagnostics.png", show))

    elif task in CLASSIFICATION_TASKS and y_true is not None:
        from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(model.pipeline, x, y_true, ax=ax, colorbar=False)
        ax.set_title("Confusion matrix")
        saved.append(_finish(fig, output_dir, "confusion_matrix.png", show))

        if task == BINARY_CLASSIFICATION:
            try:
                fig, ax = plt.subplots(figsize=(6, 5))
                RocCurveDisplay.from_estimator(model.pipeline, x, y_true, ax=ax)
                ax.plot([0, 1], [0, 1], "k--", linewidth=1)
                ax.set_title("ROC curve")
                saved.append(_finish(fig, output_dir, "roc_curve.png", show))
            except Exception:
                pass

    elif task == CLUSTERING:
        from sklearn.decomposition import PCA

        transformed = model.pipeline.named_steps["preprocessor"].transform(x)
        estimator = model.pipeline.named_steps["model"]
        labels = getattr(estimator, "labels_", None)
        if labels is None or len(labels) != len(transformed):
            labels = estimator.predict(transformed)
        coords = PCA(n_components=2).fit_transform(np.asarray(transformed))
        fig, ax = plt.subplots(figsize=(6.5, 5))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", alpha=0.75, edgecolor="none")
        ax.set(xlabel="PC 1", ylabel="PC 2", title="Clusters (PCA projection)")
        fig.colorbar(scatter, ax=ax, label="cluster")
        saved.append(_finish(fig, output_dir, "clusters.png", show))

    return [path for path in saved if path]
