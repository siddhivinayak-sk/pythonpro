"""Scripted end-to-end demonstrations using the bundled sample datasets.

Run with:  python -m main.ml.testbed.demo          (all scenarios)
           python -m main.ml.testbed.demo regression
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Dict

from .config import ExperimentConfig
from .experiment import load_model, predict_frame, run_experiment, save_model
from .dataset import load_dataset
from .registry import (
    BINARY_CLASSIFICATION,
    CLUSTERING,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)

ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "data"
MODEL_DIR = ML_DIR / "saved_models"


def _header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _run(config: ExperimentConfig, model_name: str, cv: int = 0) -> None:
    model = run_experiment(config, cross_validation_folds=cv)
    print(model.summary())
    print()
    print(model.evaluation.format())

    saved = save_model(model, str(MODEL_DIR / model_name))
    print(f"\nmodel saved -> {saved}")

    reloaded = load_model(str(saved))
    frame = load_dataset(config).sample(n=5, random_state=0)
    try:
        scored = predict_frame(reloaded, frame)
        print("\nPredictions from the reloaded model:")
        print(scored.tail(5).to_string(index=False))
    except ValueError as exc:
        print(f"\nPrediction skipped: {exc}")


def demo_regression() -> None:
    _header("1. Regression - predict the rent of a property (home-rental.csv)")
    _run(
        ExperimentConfig(
            csv_path=str(DATA_DIR / "home-rental.csv"),
            algorithm="random_forest_regressor",
            task=REGRESSION,
            feature_columns=["size", "bedrooms", "postal_code"],
            label_column="rent_amount",
            test_size=0.25,
            hyperparameters={"n_estimators": 200, "random_state": 42},
        ),
        "demo_regression_random_forest",
        cv=5,
    )


def demo_binary_classification() -> None:
    _header("2. Binary classification - diabetic yes/no (diabetes.csv)")
    _run(
        ExperimentConfig(
            csv_path=str(DATA_DIR / "diabetes.csv"),
            algorithm="gradient_boosting_classifier",
            task=BINARY_CLASSIFICATION,
            feature_columns=[
                "Pregnancies",
                "PlasmaGlucose",
                "DiastolicBloodPressure",
                "TricepsThickness",
                "SerumInsulin",
                "BMI",
                "DiabetesPedigree",
                "Age",
            ],
            label_column="Diabetic",
            test_size=0.2,
            hyperparameters={"random_state": 42},
        ),
        "demo_binary_gradient_boosting",
    )


def demo_multiclass_classification() -> None:
    _header("3. Multi-class classification - penguin species (penguins.csv)")
    _run(
        ExperimentConfig(
            csv_path=str(DATA_DIR / "penguins.csv"),
            algorithm="logistic_regression",
            task=MULTICLASS_CLASSIFICATION,
            feature_columns=["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"],
            label_column="Species",
            test_size=0.3,
        ),
        "demo_multiclass_logistic_regression",
    )


def demo_clustering() -> None:
    _header("4. Clustering - customer segmentation (customers.csv)")
    _run(
        ExperimentConfig(
            csv_path=str(DATA_DIR / "customers.csv"),
            algorithm="kmeans",
            task=CLUSTERING,
            feature_columns=["AverageSpend", "AverageFrequency"],
            hyperparameters={"n_clusters": 3, "n_init": 10, "random_state": 42},
        ),
        "demo_clustering_kmeans",
    )


def generate_shape_images(root: Path, per_class: int = 60, size: int = 64, seed: int = 7) -> Path:
    """Create a tiny circle / square / triangle dataset so the image mode is demoable offline."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    classes = ("circle", "square", "triangle")
    if root.is_dir() and all((root / name).is_dir() for name in classes):
        return root

    for name in classes:
        (root / name).mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            image = Image.new("RGB", (size, size), (245, 245, 245))
            draw = ImageDraw.Draw(image)
            margin = rng.randint(6, 14)
            box = (margin, margin, size - margin, size - margin)
            colour = (rng.randint(0, 120), rng.randint(0, 120), rng.randint(120, 255))
            if name == "circle":
                draw.ellipse(box, fill=colour)
            elif name == "square":
                draw.rectangle(box, fill=colour)
            else:
                draw.polygon(
                    [(size // 2, box[1]), (box[0], box[3]), (box[2], box[3])],
                    fill=colour,
                )
            image.save(root / name / f"{name}_{index:03d}.png")
    return root


def demo_image_classification() -> None:
    _header("5. Image classification - circle vs square vs triangle (generated images)")
    root = generate_shape_images(DATA_DIR / "shapes")
    print(f"image folder: {root}")
    _run(
        ExperimentConfig(
            csv_path=str(root),
            algorithm="svc",
            task=MULTICLASS_CLASSIFICATION,
            feature_columns=["image_path"],
            label_column="label",
            test_size=0.3,
            input_mode="image",
            image_size=(32, 32),
            image_color="gray",
            image_featurizer="flatten",
            hyperparameters={"probability": True, "random_state": 42},
        ),
        "demo_image_svc",
    )


DEMOS: Dict[str, Callable[[], None]] = {    "regression": demo_regression,
    "binary": demo_binary_classification,
    "multiclass": demo_multiclass_classification,
    "clustering": demo_clustering,
    "image": demo_image_classification,
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = argv or list(DEMOS)
    unknown = [name for name in selected if name not in DEMOS]
    if unknown:
        print(f"Unknown demo(s): {unknown}. Available: {list(DEMOS)}")
        return 1
    for name in selected:
        DEMOS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
