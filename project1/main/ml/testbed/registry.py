"""Catalog of the ML tasks and the scikit-learn (plus optional third-party) algorithms."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SUPERVISED = "supervised"
UNSUPERVISED = "unsupervised"

REGRESSION = "regression"
BINARY_CLASSIFICATION = "binary_classification"
MULTICLASS_CLASSIFICATION = "multiclass_classification"
CLUSTERING = "clustering"

TASKS: Dict[str, Dict[str, str]] = {
    REGRESSION: {"label": "Regression", "category": SUPERVISED},
    BINARY_CLASSIFICATION: {"label": "Classification (binary)", "category": SUPERVISED},
    MULTICLASS_CLASSIFICATION: {"label": "Classification (multi-class)", "category": SUPERVISED},
    CLUSTERING: {"label": "Clustering", "category": UNSUPERVISED},
}

CATEGORY_OF_TASK: Dict[str, str] = {key: meta["category"] for key, meta in TASKS.items()}

CLASSIFICATION_TASKS = (BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION)

# Clusterers that only label the rows they were fitted on.
FIT_ONLY_CLUSTERERS = {"agglomerative", "dbscan", "optics", "spectral", "hdbscan"}


@dataclass(frozen=True)
class ModelSpec:
    """Describes one selectable algorithm and how to instantiate it."""

    key: str
    label: str
    tasks: tuple
    module: str
    class_name: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    scaler: str = "standard"  # "standard" | "minmax" | "none"
    notes: str = ""

    @property
    def category(self) -> str:
        return CATEGORY_OF_TASK[self.tasks[0]]

    def is_installed(self) -> bool:
        try:
            module = importlib.import_module(self.module)
        except Exception:  # optional dependency not present
            return False
        return hasattr(module, self.class_name)

    def build(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
        params = dict(self.defaults)
        params.update(hyperparameters or {})
        estimator_cls = getattr(importlib.import_module(self.module), self.class_name)
        return estimator_cls(**params)


_CLS = CLASSIFICATION_TASKS
_REG = (REGRESSION,)
_CLU = (CLUSTERING,)

_SPECS: List[ModelSpec] = [
    # ------------------------------------------------------------------ regression
    ModelSpec("linear_regression", "Linear Regression", _REG, "sklearn.linear_model", "LinearRegression"),
    ModelSpec("ridge", "Ridge Regression", _REG, "sklearn.linear_model", "Ridge"),
    ModelSpec("lasso", "Lasso Regression", _REG, "sklearn.linear_model", "Lasso"),
    ModelSpec("elastic_net", "Elastic Net", _REG, "sklearn.linear_model", "ElasticNet"),
    ModelSpec("bayesian_ridge", "Bayesian Ridge", _REG, "sklearn.linear_model", "BayesianRidge"),
    ModelSpec("huber", "Huber Regressor (robust)", _REG, "sklearn.linear_model", "HuberRegressor"),
    ModelSpec("ransac", "RANSAC Regressor (robust)", _REG, "sklearn.linear_model", "RANSACRegressor"),
    ModelSpec("sgd_regressor", "SGD Regressor", _REG, "sklearn.linear_model", "SGDRegressor"),
    ModelSpec("knn_regressor", "K-Nearest Neighbours Regressor", _REG, "sklearn.neighbors", "KNeighborsRegressor"),
    ModelSpec("svr", "Support Vector Regressor (RBF)", _REG, "sklearn.svm", "SVR"),
    ModelSpec("linear_svr", "Linear SVR", _REG, "sklearn.svm", "LinearSVR"),
    ModelSpec("decision_tree_regressor", "Decision Tree Regressor", _REG, "sklearn.tree", "DecisionTreeRegressor"),
    ModelSpec("random_forest_regressor", "Random Forest Regressor", _REG, "sklearn.ensemble", "RandomForestRegressor"),
    ModelSpec("extra_trees_regressor", "Extra Trees Regressor", _REG, "sklearn.ensemble", "ExtraTreesRegressor"),
    ModelSpec("gradient_boosting_regressor", "Gradient Boosting Regressor", _REG, "sklearn.ensemble", "GradientBoostingRegressor"),
    ModelSpec("hist_gradient_boosting_regressor", "Histogram Gradient Boosting Regressor", _REG, "sklearn.ensemble", "HistGradientBoostingRegressor"),
    ModelSpec("adaboost_regressor", "AdaBoost Regressor", _REG, "sklearn.ensemble", "AdaBoostRegressor"),
    ModelSpec("bagging_regressor", "Bagging Regressor", _REG, "sklearn.ensemble", "BaggingRegressor"),
    ModelSpec("mlp_regressor", "Neural Network (MLP) Regressor", _REG, "sklearn.neural_network", "MLPRegressor", {"max_iter": 800}),
    ModelSpec("gaussian_process_regressor", "Gaussian Process Regressor", _REG, "sklearn.gaussian_process", "GaussianProcessRegressor", notes="Slow on large datasets."),
    ModelSpec("kernel_ridge", "Kernel Ridge Regression", _REG, "sklearn.kernel_ridge", "KernelRidge"),
    # -------------------------------------------------------------- classification
    ModelSpec("logistic_regression", "Logistic Regression", _CLS, "sklearn.linear_model", "LogisticRegression", {"max_iter": 1000}),
    ModelSpec("ridge_classifier", "Ridge Classifier", _CLS, "sklearn.linear_model", "RidgeClassifier"),
    ModelSpec("sgd_classifier", "SGD Classifier", _CLS, "sklearn.linear_model", "SGDClassifier"),
    ModelSpec("perceptron", "Perceptron", _CLS, "sklearn.linear_model", "Perceptron"),
    ModelSpec("passive_aggressive", "Passive Aggressive Classifier", _CLS, "sklearn.linear_model", "PassiveAggressiveClassifier"),
    ModelSpec("gaussian_nb", "Gaussian Naive Bayes", _CLS, "sklearn.naive_bayes", "GaussianNB"),
    ModelSpec("bernoulli_nb", "Bernoulli Naive Bayes", _CLS, "sklearn.naive_bayes", "BernoulliNB"),
    ModelSpec("multinomial_nb", "Multinomial Naive Bayes", _CLS, "sklearn.naive_bayes", "MultinomialNB", scaler="minmax", notes="Requires non-negative features, min-max scaling is applied."),
    ModelSpec("knn_classifier", "K-Nearest Neighbours Classifier", _CLS, "sklearn.neighbors", "KNeighborsClassifier"),
    ModelSpec("svc", "Support Vector Classifier (RBF)", _CLS, "sklearn.svm", "SVC", {"probability": True}),
    ModelSpec("linear_svc", "Linear SVC", _CLS, "sklearn.svm", "LinearSVC"),
    ModelSpec("decision_tree_classifier", "Decision Tree Classifier", _CLS, "sklearn.tree", "DecisionTreeClassifier"),
    ModelSpec("random_forest_classifier", "Random Forest Classifier", _CLS, "sklearn.ensemble", "RandomForestClassifier"),
    ModelSpec("extra_trees_classifier", "Extra Trees Classifier", _CLS, "sklearn.ensemble", "ExtraTreesClassifier"),
    ModelSpec("gradient_boosting_classifier", "Gradient Boosting Classifier", _CLS, "sklearn.ensemble", "GradientBoostingClassifier"),
    ModelSpec("hist_gradient_boosting_classifier", "Histogram Gradient Boosting Classifier", _CLS, "sklearn.ensemble", "HistGradientBoostingClassifier"),
    ModelSpec("adaboost_classifier", "AdaBoost Classifier", _CLS, "sklearn.ensemble", "AdaBoostClassifier"),
    ModelSpec("bagging_classifier", "Bagging Classifier", _CLS, "sklearn.ensemble", "BaggingClassifier"),
    ModelSpec("mlp_classifier", "Neural Network (MLP) Classifier", _CLS, "sklearn.neural_network", "MLPClassifier", {"max_iter": 800}),
    ModelSpec("lda", "Linear Discriminant Analysis", _CLS, "sklearn.discriminant_analysis", "LinearDiscriminantAnalysis"),
    ModelSpec("qda", "Quadratic Discriminant Analysis", _CLS, "sklearn.discriminant_analysis", "QuadraticDiscriminantAnalysis"),
    ModelSpec("gaussian_process_classifier", "Gaussian Process Classifier", _CLS, "sklearn.gaussian_process", "GaussianProcessClassifier", notes="Slow on large datasets."),
    # ------------------------------------------------------------------ clustering
    ModelSpec("kmeans", "K-Means", _CLU, "sklearn.cluster", "KMeans", {"n_clusters": 3, "n_init": 10}),
    ModelSpec("minibatch_kmeans", "Mini-Batch K-Means", _CLU, "sklearn.cluster", "MiniBatchKMeans", {"n_clusters": 3, "n_init": 10}),
    ModelSpec("bisecting_kmeans", "Bisecting K-Means", _CLU, "sklearn.cluster", "BisectingKMeans", {"n_clusters": 3}),
    ModelSpec("agglomerative", "Agglomerative (hierarchical) Clustering", _CLU, "sklearn.cluster", "AgglomerativeClustering", {"n_clusters": 3}, notes="No predict(): cannot score unseen rows."),
    ModelSpec("dbscan", "DBSCAN", _CLU, "sklearn.cluster", "DBSCAN", notes="Density based, picks the cluster count itself. No predict()."),
    ModelSpec("optics", "OPTICS", _CLU, "sklearn.cluster", "OPTICS", notes="Density based. No predict()."),
    ModelSpec("hdbscan", "HDBSCAN", _CLU, "sklearn.cluster", "HDBSCAN", notes="Needs scikit-learn >= 1.3."),
    ModelSpec("birch", "BIRCH", _CLU, "sklearn.cluster", "Birch", {"n_clusters": 3}),
    ModelSpec("mean_shift", "Mean Shift", _CLU, "sklearn.cluster", "MeanShift"),
    ModelSpec("spectral", "Spectral Clustering", _CLU, "sklearn.cluster", "SpectralClustering", {"n_clusters": 3}, notes="No predict()."),
    ModelSpec("affinity_propagation", "Affinity Propagation", _CLU, "sklearn.cluster", "AffinityPropagation"),
    ModelSpec("gaussian_mixture", "Gaussian Mixture Model", _CLU, "sklearn.mixture", "GaussianMixture", {"n_components": 3}),
    # ------------------------------- optional third-party gradient boosting engines
    ModelSpec("xgboost_regressor", "XGBoost Regressor", _REG, "xgboost", "XGBRegressor"),
    ModelSpec("xgboost_classifier", "XGBoost Classifier", _CLS, "xgboost", "XGBClassifier"),
    ModelSpec("lightgbm_regressor", "LightGBM Regressor", _REG, "lightgbm", "LGBMRegressor", {"verbose": -1}),
    ModelSpec("lightgbm_classifier", "LightGBM Classifier", _CLS, "lightgbm", "LGBMClassifier", {"verbose": -1}),
    ModelSpec("catboost_regressor", "CatBoost Regressor", _REG, "catboost", "CatBoostRegressor", {"verbose": 0}),
    ModelSpec("catboost_classifier", "CatBoost Classifier", _CLS, "catboost", "CatBoostClassifier", {"verbose": 0}),
]

REGISTRY: Dict[str, ModelSpec] = {spec.key: spec for spec in _SPECS if spec.is_installed()}


def tasks_for_category(category: str) -> List[str]:
    return [key for key, meta in TASKS.items() if meta["category"] == category]


def available_algorithms(task: str) -> List[ModelSpec]:
    """All installed algorithms that can be used for the given task."""
    if task not in TASKS:
        raise KeyError(f"Unknown task '{task}'. Known tasks: {sorted(TASKS)}")
    return sorted((spec for spec in REGISTRY.values() if task in spec.tasks), key=lambda s: s.label)


def get_algorithm(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown algorithm '{key}'. Known algorithms: {sorted(REGISTRY)}") from exc
