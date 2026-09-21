"""Command line front end for the ML testbed: interactive wizard plus scriptable commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

from .config import ExperimentConfig
from .dataset import infer_task, load_csv, load_dataset, profile_columns
from .experiment import load_model, predict_csv, run_experiment, save_model
from .images import FEATURIZERS, IMAGE_LABEL_COLUMN, IMAGE_PATH_COLUMN, load_images
from .registry import (
    CLUSTERING,
    SUPERVISED,
    TASKS,
    UNSUPERVISED,
    available_algorithms,
    tasks_for_category,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"


# --------------------------------------------------------------------- helpers
def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _choose(prompt: str, options: Sequence[str], descriptions: Optional[Sequence[str]] = None, default: int = 1) -> int:
    descriptions = descriptions or options
    print(f"\n{prompt}")
    for index, text in enumerate(descriptions, start=1):
        print(f"  [{index:>2}] {text}")
    while True:
        raw = input(f"Choice [1-{len(options)}] (default {default}): ").strip()
        if not raw:
            return default - 1
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("  Invalid choice, try again.")


def _choose_many(prompt: str, options: Sequence[str], default: Sequence[int]) -> List[str]:
    print(f"\n{prompt}")
    for index, text in enumerate(options, start=1):
        print(f"  [{index:>2}] {text}")
    default_text = ",".join(str(i + 1) for i in default)
    while True:
        raw = input(f"Comma separated numbers, 'all' (default {default_text}): ").strip()
        if not raw:
            return [options[i] for i in default]
        if raw.lower() == "all":
            return list(options)
        try:
            picks = [int(part) - 1 for part in raw.split(",") if part.strip()]
        except ValueError:
            print("  Numbers only, try again.")
            continue
        if picks and all(0 <= p < len(options) for p in picks):
            return [options[p] for p in picks]
        print("  Invalid selection, try again.")


def _ask(prompt: str, default: str = "") -> str:
    raw = input(f"{prompt}{f' (default {default})' if default else ''}: ").strip()
    return raw or default


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def _sample_datasets() -> List[Path]:
    return sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.is_dir() else []


# ----------------------------------------------------------------- list command
def list_catalog(task: Optional[str] = None) -> None:
    tasks = [task] if task else list(TASKS)
    for key in tasks:
        meta = TASKS[key]
        _banner(f"{meta['label']}  ({meta['category']}, task key = {key})")
        for spec in available_algorithms(key):
            note = f"  # {spec.notes}" if spec.notes else ""
            print(f"  {spec.key:<38} {spec.label}{note}")


# ---------------------------------------------------------------- train command
def report(model, show_plots: bool = False, plot_dir: Optional[str] = None) -> None:
    _banner("Training finished")
    print(model.summary())
    if model.evaluation:
        print()
        print(model.evaluation.format())
    if plot_dir or show_plots:
        from .plots import plot_results

        frame = load_dataset(model.config)
        x = frame[model.feature_columns]
        y = frame[model.config.label_column] if model.config.label_column else None
        saved = plot_results(model, x, y, output_dir=plot_dir, show=show_plots)
        for path in saved:
            print(f"chart saved -> {path}")


def train_command(args: argparse.Namespace) -> None:
    if args.input_mode == "image":
        config = _image_config_from_args(args)
    else:
        frame = load_csv(args.csv, args.separator)
        features = args.features or [c for c in frame.columns if c != args.label]
        task = args.task or infer_task(frame, args.label)
        config = ExperimentConfig(
            csv_path=args.csv,
            algorithm=args.algorithm,
            task=task,
            feature_columns=features,
            label_column=args.label,
            test_size=args.test_size,
            random_state=args.random_state,
            hyperparameters=json.loads(args.params) if args.params else {},
            csv_separator=args.separator,
        )
    model = run_experiment(config, cross_validation_folds=args.cv)
    report(model, show_plots=False, plot_dir=args.plots)
    if args.save:
        path = save_model(model, args.save)
        print(f"\nmodel saved -> {path}")


def _image_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    label = None if args.task == CLUSTERING else (args.label or IMAGE_LABEL_COLUMN)
    probe = ExperimentConfig(
        csv_path=args.csv,
        algorithm=args.algorithm,
        task=args.task or CLUSTERING,
        feature_columns=[IMAGE_PATH_COLUMN],
        label_column=None if (args.task or CLUSTERING) == CLUSTERING else label,
        input_mode="image",
        csv_separator=args.separator,
    )
    frame = load_dataset(probe)
    task = args.task or infer_task(frame, label)
    return ExperimentConfig(
        csv_path=args.csv,
        algorithm=args.algorithm,
        task=task,
        feature_columns=[IMAGE_PATH_COLUMN],
        label_column=None if task == CLUSTERING else label,
        test_size=args.test_size,
        random_state=args.random_state,
        hyperparameters=json.loads(args.params) if args.params else {},
        csv_separator=args.separator,
        input_mode="image",
        image_size=(args.image_size, args.image_size),
        image_color=args.image_color,
        image_featurizer=args.featurizer,
    )


# -------------------------------------------------------------- predict command
def predict_command(args: argparse.Namespace) -> None:
    scored = predict_csv(args.model, args.csv, args.output, args.separator)
    _banner(f"Predictions ({len(scored)} rows)")
    print(scored.head(args.head).to_string(index=False))
    if args.output:
        print(f"\npredictions written -> {args.output}")


# ------------------------------------------------------------ interactive wizard
def interactive() -> None:
    _banner("Machine Learning Testbed")
    print("Train, evaluate, save and reuse models from a CSV file or a folder of images.")

    input_mode = ["tabular", "image"][
        _choose(
            "Input type:",
            ["tabular", "image"],
            ["tabular data from a CSV file", "images from a folder-per-class directory or a manifest CSV"],
        )
    ]

    if input_mode == "image":
        _interactive_images()
        return

    samples = _sample_datasets()
    if samples:
        labels = [f"{p.name}" for p in samples] + ["<enter another path>"]
        index = _choose("Pick a dataset:", labels)
        csv_path = str(samples[index]) if index < len(samples) else _ask("Path to CSV")
    else:
        csv_path = _ask("Path to CSV")

    separator = _ask("CSV separator", ",")
    frame = load_csv(csv_path, separator)

    _banner(f"{Path(csv_path).name}: {len(frame)} rows x {len(frame.columns)} columns")
    for profile in profile_columns(frame):
        print("  " + profile.describe())

    categories = [SUPERVISED, UNSUPERVISED]
    category = categories[
        _choose(
            "Model category:",
            categories,
            ["supervised (you have a label column)", "unsupervised (no label, find structure)"],
        )
    ]

    task_keys = tasks_for_category(category)
    columns = list(frame.columns)

    label_column: Optional[str] = None
    if category == SUPERVISED:
        label_column = columns[_choose("Label (target) column:", columns, default=len(columns))]
        suggested = infer_task(frame, label_column)
        default_index = task_keys.index(suggested) + 1 if suggested in task_keys else 1
        task = task_keys[
            _choose(
                f"Task type (auto-detected: {TASKS[suggested]['label']}):",
                task_keys,
                [TASKS[k]["label"] for k in task_keys],
                default=default_index,
            )
        ]
    else:
        task = task_keys[_choose("Task type:", task_keys, [TASKS[k]["label"] for k in task_keys])]
        if _ask_yes_no("Do you have a ground-truth label column to score the clusters against?", False):
            label_column = columns[_choose("Ground-truth column:", columns, default=len(columns))]

    candidate_features = [c for c in columns if c != label_column]
    features = _choose_many(
        "Feature columns:",
        candidate_features,
        default=list(range(len(candidate_features))),
    )

    specs = available_algorithms(task)
    spec = specs[
        _choose(
            "Algorithm:",
            [s.key for s in specs],
            [f"{s.label:<45} ({s.key}){'  # ' + s.notes if s.notes else ''}" for s in specs],
        )
    ]

    test_size = 0.2
    if task != CLUSTERING:
        test_size = float(_ask("Validation split fraction", "0.2"))

    hyperparameters = {}
    if spec.defaults:
        print(f"\nDefault hyper-parameters for {spec.label}: {spec.defaults}")
    raw_params = _ask("Hyper-parameter overrides as JSON (blank to skip)")
    if raw_params:
        hyperparameters = json.loads(raw_params)

    config = ExperimentConfig(
        csv_path=csv_path,
        algorithm=spec.key,
        task=task,
        feature_columns=features,
        label_column=label_column,
        test_size=test_size,
        hyperparameters=hyperparameters,
        csv_separator=separator,
    )
    _train_and_offer_to_save(config, spec)


def _interactive_images() -> None:
    source = _ask("Image folder (one sub-folder per class) or manifest CSV")
    frame = load_images(source)
    labelled = IMAGE_LABEL_COLUMN in frame.columns

    _banner(f"{len(frame)} images found")
    if labelled:
        print(frame[IMAGE_LABEL_COLUMN].value_counts().to_string())
    print(f"\nexample: {frame[IMAGE_PATH_COLUMN].iloc[0]}")

    categories = [SUPERVISED, UNSUPERVISED]
    default_category = 1 if labelled else 2
    category = categories[
        _choose(
            "Model category:",
            categories,
            ["supervised (classify the images by their folder name)", "unsupervised (group similar images)"],
            default=default_category,
        )
    ]

    label_column: Optional[str] = None
    if category == SUPERVISED:
        if not labelled:
            raise ValueError("Supervised image training needs labels: use a folder-per-class layout.")
        label_column = IMAGE_LABEL_COLUMN
        task = infer_task(frame, label_column)
        print(f"\nDetected task: {TASKS[task]['label']} ({frame[label_column].nunique()} classes)")
    else:
        task = CLUSTERING
        if labelled and _ask_yes_no("Score the clusters against the folder labels?", True):
            label_column = IMAGE_LABEL_COLUMN

    featurizer_keys = sorted(FEATURIZERS)
    featurizer = featurizer_keys[
        _choose(
            "Image featurizer:",
            featurizer_keys,
            [f"{key:<10} {FEATURIZERS[key]}" for key in featurizer_keys],
            default=featurizer_keys.index("flatten") + 1,
        )
    ]
    size = int(_ask("Resize images to N x N pixels", "32"))
    color = ["gray", "rgb"][_choose("Colour mode:", ["gray", "rgb"], ["grayscale (fewer features)", "RGB"])]

    specs = available_algorithms(task)
    spec = specs[
        _choose(
            "Algorithm:",
            [s.key for s in specs],
            [f"{s.label:<45} ({s.key}){'  # ' + s.notes if s.notes else ''}" for s in specs],
        )
    ]

    test_size = 0.2 if task == CLUSTERING else float(_ask("Validation split fraction", "0.2"))
    if spec.defaults:
        print(f"\nDefault hyper-parameters for {spec.label}: {spec.defaults}")
    raw_params = _ask("Hyper-parameter overrides as JSON (blank to skip)")

    config = ExperimentConfig(
        csv_path=source,
        algorithm=spec.key,
        task=task,
        feature_columns=[IMAGE_PATH_COLUMN],
        label_column=label_column,
        test_size=test_size,
        hyperparameters=json.loads(raw_params) if raw_params else {},
        input_mode="image",
        image_size=(size, size),
        image_color=color,
        image_featurizer=featurizer,
    )
    _train_and_offer_to_save(config, spec)


def _train_and_offer_to_save(config: ExperimentConfig, spec) -> None:
    folds = 0
    if config.task != CLUSTERING and _ask_yes_no("Run cross-validation as well?", False):
        folds = int(_ask("Number of folds", "5"))

    print("\nTraining...")
    model = run_experiment(config, cross_validation_folds=folds)
    report(model)

    if _ask_yes_no("Generate charts?", True):
        plot_dir = _ask("Chart output folder", str(DEFAULT_MODEL_DIR / "charts"))
        report(model, show_plots=False, plot_dir=plot_dir)

    if _ask_yes_no("Save this model?", True):
        default_path = str(DEFAULT_MODEL_DIR / f"{Path(config.csv_path).stem}_{spec.key}.joblib")
        saved = save_model(model, _ask("Model file", default_path))
        print(f"model saved -> {saved}")

        if _ask_yes_no("Score new data with the saved model now?", False):
            source = _ask("Data to score", config.csv_path)
            output = _ask("Write predictions to (blank to only print)")
            scored = predict_csv(str(saved), source, output or None, config.csv_separator)
            print(scored.head(10).to_string(index=False))

    print("\nDone.")


# ------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-testbed", description="Machine learning testbed for CSV data.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("interactive", help="Guided wizard (default when no command is given).")

    list_parser = sub.add_parser("list", help="List the available tasks and algorithms.")
    list_parser.add_argument("--task", choices=sorted(TASKS), help="Restrict the listing to one task.")

    train_parser = sub.add_parser("train", help="Train and evaluate a model non-interactively.")
    train_parser.add_argument("--csv", required=True, help="CSV file, or (image mode) an image folder / manifest CSV.")
    train_parser.add_argument("--algorithm", required=True)
    train_parser.add_argument("--label", help="Target column (omit for clustering).")
    train_parser.add_argument("--features", nargs="*", help="Feature columns (default: every other column).")
    train_parser.add_argument("--task", choices=sorted(TASKS), help="Override the auto-detected task.")
    train_parser.add_argument("--input-mode", choices=["tabular", "image"], default="tabular")
    train_parser.add_argument("--image-size", type=int, default=32, help="Images are resized to NxN pixels.")
    train_parser.add_argument("--image-color", choices=["gray", "rgb"], default="gray")
    train_parser.add_argument("--featurizer", choices=sorted(FEATURIZERS), default="flatten")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--cv", type=int, default=0, help="Cross-validation folds (0 disables).")
    train_parser.add_argument("--params", help='Hyper-parameters as JSON, e.g. \'{"n_estimators": 300}\'')
    train_parser.add_argument("--separator", default=",")
    train_parser.add_argument("--save", help="Path to store the fitted model.")
    train_parser.add_argument("--plots", help="Folder to write evaluation charts into.")

    predict_parser = sub.add_parser("predict", help="Score a CSV (or image folder) with a saved model.")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--csv", required=True, help="CSV file, or (image models) an image folder / manifest.")
    predict_parser.add_argument("--output", help="Write the scored rows to this CSV.")
    predict_parser.add_argument("--separator", default=",")
    predict_parser.add_argument("--head", type=int, default=10)

    info_parser = sub.add_parser("info", help="Show the metadata of a saved model.")
    info_parser.add_argument("--model", required=True)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "interactive"

    if command == "interactive":
        interactive()
    elif command == "list":
        list_catalog(getattr(args, "task", None))
    elif command == "train":
        train_command(args)
    elif command == "predict":
        predict_command(args)
    elif command == "info":
        model = load_model(args.model)
        print(model.summary())
        if model.evaluation and model.evaluation.metrics:
            print()
            print(model.evaluation.format())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
