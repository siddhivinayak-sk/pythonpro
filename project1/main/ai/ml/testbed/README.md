# ML Testbed

A small, demo-friendly bench for training, evaluating, saving and reusing machine learning
models from a CSV file **or a folder of images**.

## What it does

| Step | Where |
|------|-------|
| Pick a CSV (or image folder) and inspect it | [dataset.py](dataset.py), [images.py](images.py) |
| Choose category (supervised / unsupervised) and task (regression, binary / multi-class classification, clustering) | [registry.py](registry.py) |
| Choose label + feature columns and the train/validation split | [config.py](config.py) |
| Pick any installed algorithm for that task | [registry.py](registry.py) |
| Impute, scale and one-hot encode automatically (or featurise images) | [preprocessing.py](preprocessing.py), [images.py](images.py) |
| Run the evaluation designed for that task | [evaluation.py](evaluation.py) |
| Save the model, reload it and predict | [experiment.py](experiment.py) |
| Task-appropriate charts | [plots.py](plots.py) |
| Interactive wizard + scriptable commands | [cli.py](cli.py) |

## Requirements

```powershell
pip install pandas scikit-learn joblib matplotlib
# image mode
pip install pillow          # required
pip install scikit-image    # only for the HOG featurizer
# optional extra algorithms, auto-detected when installed
pip install xgboost lightgbm catboost
```

## Run it

All commands are executed from the `project1` folder.

```powershell
# guided wizard - the easiest way to demo
python -m main.ml.testbed

# what can I train?
python -m main.ml.testbed list
python -m main.ml.testbed list --task clustering

# non-interactive training
python -m main.ml.testbed train --csv main/ml/data/diabetes.csv --label Diabetic `
    --algorithm random_forest_classifier --test-size 0.2 --cv 5 `
    --save main/ml/saved_models/diabetes_rf.joblib --plots main/ml/saved_models/charts

# clustering (no label column)
python -m main.ml.testbed train --csv main/ml/data/customers.csv --task clustering `
    --algorithm kmeans --features AverageSpend AverageFrequency `
    --params '{\"n_clusters\": 3}' --save main/ml/saved_models/customers_kmeans.joblib

# reuse a saved model
python -m main.ml.testbed info --model main/ml/saved_models/diabetes_rf.joblib
python -m main.ml.testbed predict --model main/ml/saved_models/diabetes_rf.joblib `
    --csv main/ml/data/diabetes.csv --output main/ml/saved_models/scored.csv

# four scripted end-to-end scenarios
python -m main.ml.testbed.demo
python -m main.ml.testbed.demo clustering
```

## Image mode

Images are resized to a fixed size and turned into a numeric feature vector, so **every**
classifier and clusterer in the catalog works on them unchanged.

Layout — one sub-folder per class (the folder name becomes the label):

```
shapes/
  circle/   circle_000.png ...
  square/   square_000.png ...
  triangle/ triangle_000.png ...
```

A manifest CSV with `image_path` (and optionally `label`) columns works too; relative paths are
resolved against the CSV's folder.

Featurizers (`--featurizer`):

| Key | What it produces |
|-----|------------------|
| `flatten` | Resized raw pixels scaled to `[0, 1]` — the default |
| `hog` | Histogram of Oriented Gradients, shape-sensitive (needs `scikit-image`) |
| `stats` | Per-channel histogram + intensity statistics — compact and fast |

```powershell
# generates a sample circle/square/triangle dataset, then trains and scores it
python -m main.ml.testbed.demo image

# classify images from a folder
python -m main.ml.testbed train --input-mode image --csv main/ml/data/shapes `
    --algorithm svc --featurizer hog --image-size 48 --image-color gray `
    --save main/ml/saved_models/shapes_svc.joblib --plots main/ml/saved_models/charts

# group images without labels
python -m main.ml.testbed train --input-mode image --csv main/ml/data/shapes `
    --task clustering --algorithm kmeans --featurizer stats --params '{\"n_clusters\": 3}'

# score a folder of images with a saved model
python -m main.ml.testbed predict --model main/ml/saved_models/shapes_svc.joblib `
    --csv main/ml/data/shapes --output main/ml/saved_models/image_scores.csv
```

From Python:

```python
from main.ml.testbed import image_config, run_experiment, load_model, predict_images

config = image_config(
    source="main/ml/data/shapes",
    algorithm="random_forest_classifier",
    task="multiclass_classification",
    image_size=(32, 32),
    image_color="gray",
    image_featurizer="hog",
)
model = run_experiment(config)
print(model.evaluation.format())

predict_images(model, ["some/photo.png", "another/photo.png"])
```

Note that classical estimators on pixel/HOG features suit small, clean image sets. For photographic
data, use a CNN or feed pretrained CNN embeddings in as features.

## Use it from Python

```python
from main.ml.testbed import ExperimentConfig, run_experiment, save_model, load_model, predict_frame

config = ExperimentConfig(
    csv_path="main/ml/data/penguins.csv",
    algorithm="random_forest_classifier",
    task="multiclass_classification",
    feature_columns=["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"],
    label_column="Species",
    test_size=0.3,
)

model = run_experiment(config, cross_validation_folds=5)
print(model.summary())
print(model.evaluation.format())

path = save_model(model, "main/ml/saved_models/penguins_rf.joblib")
scored = predict_frame(load_model(str(path)), new_dataframe)
```

## Evaluation per task

* **Regression** - R², explained variance, MAE, MSE, RMSE, median absolute error, max error, MAPE.
* **Binary classification** - accuracy, balanced accuracy, precision / recall / F1, Cohen's kappa,
  MCC, ROC AUC, average precision, log loss, confusion matrix, per-class report.
* **Multi-class classification** - the same plus macro and weighted F1 and one-vs-rest ROC AUC.
* **Clustering** - cluster count and sizes, silhouette, Calinski-Harabasz, Davies-Bouldin,
  inertia / BIC / AIC where applicable, and ARI, NMI, homogeneity, completeness, V-measure when a
  ground-truth column is supplied.

## Notes

* Every model is wrapped in a scikit-learn `Pipeline`, so preprocessing is saved together with the
  estimator and applied identically at prediction time.
* Saved models are `.joblib` files with a readable `.json` sidecar holding the config and metrics.
* Density-based clusterers (DBSCAN, OPTICS, HDBSCAN, spectral, agglomerative) only label the rows
  they were fitted on; the testbed reports this instead of failing.
* Algorithms from optional packages appear in the catalog only when the package is installed.
