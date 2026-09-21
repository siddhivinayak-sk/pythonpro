"""A self-contained testbed to train, evaluate, save and reuse ML models from CSV files."""

from .config import ExperimentConfig, image_config
from .evaluation import EvaluationResult, evaluate
from .experiment import (
    TrainedModel,
    load_model,
    predict_csv,
    predict_frame,
    predict_images,
    run_experiment,
    save_model,
)
from .images import ImageFeaturizer, load_image_folder, load_image_manifest, load_images
from .registry import (
    CATEGORY_OF_TASK,
    TASKS,
    ModelSpec,
    available_algorithms,
    get_algorithm,
    tasks_for_category,
)

__all__ = [
    "CATEGORY_OF_TASK",
    "EvaluationResult",
    "ExperimentConfig",
    "ImageFeaturizer",
    "ModelSpec",
    "TASKS",
    "TrainedModel",
    "available_algorithms",
    "evaluate",
    "get_algorithm",
    "image_config",
    "load_image_folder",
    "load_image_manifest",
    "load_images",
    "load_model",
    "predict_csv",
    "predict_frame",
    "predict_images",
    "run_experiment",
    "save_model",
    "tasks_for_category",
]
