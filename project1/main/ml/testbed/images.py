"""Image input support: dataset discovery, featurisation and image-specific charts.

Images are turned into fixed length numeric vectors so every classifier and clusterer
already registered in the testbed can be trained on them without further changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

IMAGE_PATH_COLUMN = "image_path"
IMAGE_LABEL_COLUMN = "label"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")

FEATURIZERS = {
    "flatten": "Resized raw pixels, scaled to [0, 1]",
    "hog": "Histogram of Oriented Gradients (needs scikit-image)",
    "stats": "Per-channel histogram + intensity statistics (compact, fast)",
}


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Image mode needs Pillow. Install it with: pip install pillow") from exc
    return Image


def load_image_folder(root: str, label_from: str = "folder") -> pd.DataFrame:
    """Build an image manifest from a directory where each sub-folder is one class."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Image folder not found: {root_path}")

    rows = []
    for path in sorted(root_path.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        label = path.parent.name if label_from == "folder" else path.stem
        rows.append({IMAGE_PATH_COLUMN: str(path), IMAGE_LABEL_COLUMN: label})

    if not rows:
        raise ValueError(f"No images with extensions {IMAGE_EXTENSIONS} found under '{root_path}'.")
    return pd.DataFrame(rows)


def load_image_manifest(
    csv_path: str,
    path_column: str = IMAGE_PATH_COLUMN,
    label_column: Optional[str] = IMAGE_LABEL_COLUMN,
    separator: str = ",",
) -> pd.DataFrame:
    """Read a CSV listing image paths (and optionally labels); paths may be relative to the CSV."""
    source = Path(csv_path).expanduser()
    frame = pd.read_csv(source, sep=separator)
    if path_column not in frame.columns:
        raise KeyError(f"Manifest '{source}' has no column '{path_column}'. Columns: {list(frame.columns)}")

    base = source.parent
    frame[IMAGE_PATH_COLUMN] = [
        str(p if (p := Path(str(value)).expanduser()).is_absolute() else (base / value).resolve())
        for value in frame[path_column]
    ]
    if label_column and label_column in frame.columns:
        frame[IMAGE_LABEL_COLUMN] = frame[label_column]
    return frame


def load_images(source: str, separator: str = ",") -> pd.DataFrame:
    """Accept either a folder-per-class directory or a manifest CSV."""
    path = Path(source).expanduser()
    if path.is_dir():
        return load_image_folder(str(path))
    if path.is_file():
        return load_image_manifest(str(path), separator=separator)
    raise FileNotFoundError(f"Image source not found: {path}")


class ImageFeaturizer(BaseEstimator, TransformerMixin):
    """Turns a column of image file paths into a fixed length numeric feature matrix."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (32, 32),
        color_mode: str = "gray",
        method: str = "flatten",
        hog_orientations: int = 9,
        hog_pixels_per_cell: Tuple[int, int] = (8, 8),
        hog_cells_per_block: Tuple[int, int] = (2, 2),
        histogram_bins: int = 16,
    ) -> None:
        self.image_size = image_size
        self.color_mode = color_mode
        self.method = method
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.histogram_bins = histogram_bins

    # ------------------------------------------------------------- sklearn API
    def fit(self, X, y=None):  # noqa: N803 - sklearn naming
        _require_pillow()
        if self.method not in FEATURIZERS:
            raise ValueError(f"Unknown featurizer '{self.method}'. Choose from {sorted(FEATURIZERS)}.")
        self.n_features_out_ = len(self._featurize(self._paths(X)[0]))
        return self

    def transform(self, X):  # noqa: N803 - sklearn naming
        _require_pillow()
        paths = self._paths(X)
        return np.vstack([self._featurize(path) for path in paths])

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _paths(X) -> List[str]:
        if isinstance(X, pd.DataFrame):
            if IMAGE_PATH_COLUMN in X.columns:
                values: Sequence = X[IMAGE_PATH_COLUMN].tolist()
            else:
                values = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            values = X.tolist()
        else:
            values = np.asarray(X, dtype=object).ravel().tolist()
        if not len(values):
            raise ValueError("No image paths were supplied.")
        return [str(value) for value in values]

    def _load_array(self, path: str) -> np.ndarray:
        Image = _require_pillow()
        with Image.open(path) as handle:
            image = handle.convert("L" if self.color_mode == "gray" else "RGB")
            image = image.resize(tuple(self.image_size))
            return np.asarray(image, dtype=np.float32) / 255.0

    def _featurize(self, path: str) -> np.ndarray:
        array = self._load_array(path)
        if self.method == "flatten":
            return array.ravel()
        if self.method == "hog":
            return self._hog(array)
        return self._stats(array)

    def _hog(self, array: np.ndarray) -> np.ndarray:
        try:
            from skimage.feature import hog
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("The 'hog' featurizer needs scikit-image: pip install scikit-image") from exc
        gray = array if array.ndim == 2 else array.mean(axis=2)
        return hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=tuple(self.hog_pixels_per_cell),
            cells_per_block=tuple(self.hog_cells_per_block),
            feature_vector=True,
        )

    def _stats(self, array: np.ndarray) -> np.ndarray:
        channels = [array] if array.ndim == 2 else [array[:, :, i] for i in range(array.shape[2])]
        features: List[np.ndarray] = []
        for channel in channels:
            histogram, _ = np.histogram(channel, bins=self.histogram_bins, range=(0.0, 1.0), density=True)
            features.append(histogram)
            features.append(
                np.array([channel.mean(), channel.std(), channel.min(), channel.max(), np.median(channel)])
            )
        return np.concatenate(features)


def plot_image_predictions(
    model,
    frame: pd.DataFrame,
    output_dir: Optional[str] = None,
    show: bool = False,
    max_images: int = 12,
) -> List[str]:
    """Show a grid of sample images with their predicted (and true) labels."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Image = _require_pillow()
    from .experiment import predict_frame

    sample = frame.sample(n=min(max_images, len(frame)), random_state=0)
    scored = predict_frame(model, sample)
    prediction_column = "cluster" if "cluster" in scored.columns else "prediction"
    truth_column = model.config.label_column

    columns = min(4, len(sample))
    rows = int(np.ceil(len(sample) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.2 * rows))
    for axis, (_, row) in zip(np.atleast_1d(axes).ravel(), scored.iterrows()):
        with Image.open(row[IMAGE_PATH_COLUMN]) as handle:
            axis.imshow(handle.convert("RGB"))
        title = f"pred: {row[prediction_column]}"
        if truth_column and truth_column in row:
            title += f"\ntrue: {row[truth_column]}"
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    for axis in np.atleast_1d(axes).ravel()[len(sample) :]:
        axis.axis("off")
    fig.tight_layout()

    saved = None
    if output_dir:
        target = Path(output_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        saved = str(target / "image_predictions.png")
        fig.savefig(saved, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return [saved] if saved else []
