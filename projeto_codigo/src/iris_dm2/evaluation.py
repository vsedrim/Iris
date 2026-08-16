from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from iris_dm2.data import (
    DatasetValidationError,
    IrisDataset,
    require_clinical_label_basis,
    require_identity_basis,
    sha256_file,
)
from iris_dm2.features import FeatureTable

MODEL_NAMES = ("logistic_regression", "svm", "random_forest", "mlp", "adaboost")
METRIC_COLUMNS = ("accuracy", "sensitivity", "specificity", "precision", "f1")


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    fold_assignments: pd.DataFrame
    summary: pd.DataFrame


def build_model(name: str, seed: int):
    if name == "logistic_regression":
        return make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
    if name == "svm":
        return make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", class_weight="balanced"),
        )
    if name == "random_forest":
        return make_pipeline(
            VarianceThreshold(),
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if name == "mlp":
        return make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-3,
                early_stopping=False,
                max_iter=600,
                random_state=seed,
            ),
        )
    if name == "adaboost":
        return make_pipeline(
            VarianceThreshold(),
            AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=seed),
        )
    raise ValueError(f"unknown model: {name}")


def create_grouped_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    assignments: list[dict[str, object]] = []
    placeholder_features = np.zeros((len(labels), 1), dtype=np.float32)
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(placeholder_features, labels, groups)
    ):
        overlap = set(groups[train_indices]) & set(groups[test_indices])
        if overlap:
            raise DatasetValidationError(
                f"person leakage in seed {seed}, fold {fold_index}: {sorted(overlap)}"
            )
        folds.append((train_indices, test_indices))
        assignments.extend(
            {
                "sample_id": str(sample_ids[index]),
                "person_id": str(groups[index]),
                "label": int(labels[index]),
                "seed": seed,
                "fold": fold_index,
            }
            for index in test_indices
        )
    return folds, pd.DataFrame(assignments)


def classification_metrics(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "sensitivity": float(recall_score(y_true, y_predicted, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_predicted, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_predicted, pos_label=1, zero_division=0)),
    }


def _feature_mask(table: FeatureTable, feature_set: str) -> np.ndarray:
    if feature_set == "all":
        return np.ones(len(table.names), dtype=bool)
    requested_groups = set(feature_set.split("+"))
    unknown = requested_groups - set(table.groups.tolist())
    if unknown:
        message = f"feature set {feature_set!r} references unavailable groups: {sorted(unknown)}"
        raise ValueError(message)
    return np.isin(table.groups, list(requested_groups))


def _validate_feature_tables(dataset: IrisDataset, tables: dict[str, FeatureTable]) -> None:
    if "original" not in tables:
        raise DatasetValidationError("an original feature table is required for model training")
    reference = tables["original"]
    for variant, table in tables.items():
        np.testing.assert_array_equal(table.sample_ids, dataset.sample_ids)
        np.testing.assert_array_equal(table.names, reference.names)
        np.testing.assert_array_equal(table.groups, reference.groups)
        if table.variant != variant:
            raise DatasetValidationError(
                f"feature table key {variant!r} does not match stored variant {table.variant!r}"
            )


def evaluate_feature_tables(
    dataset: IrisDataset,
    tables: dict[str, FeatureTable],
    model_names: tuple[str, ...],
    feature_sets: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
) -> EvaluationResult:
    _validate_feature_tables(dataset, tables)
    unknown_models = set(model_names) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"unknown models: {sorted(unknown_models)}")

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    assignment_frames: list[pd.DataFrame] = []
    original = tables["original"]

    for seed in seeds:
        folds, assignments = create_grouped_folds(
            dataset.labels,
            dataset.person_ids,
            dataset.sample_ids,
            n_splits=n_splits,
            seed=seed,
        )
        assignment_frames.append(assignments)
        for feature_set in feature_sets:
            mask = _feature_mask(original, feature_set)
            if not np.any(mask):
                raise DatasetValidationError(f"feature set {feature_set!r} selected no columns")
            for fold_index, (train_indices, test_indices) in enumerate(folds):
                for model_name in model_names:
                    model = build_model(model_name, seed)
                    start_time = perf_counter()
                    train_values = original.values[train_indices][:, mask]
                    model.fit(train_values, dataset.labels[train_indices])
                    fit_seconds = perf_counter() - start_time
                    for variant, table in tables.items():
                        predicted = model.predict(table.values[test_indices][:, mask])
                        row: dict[str, object] = {
                            "model": model_name,
                            "feature_set": feature_set,
                            "variant": variant,
                            "seed": seed,
                            "fold": fold_index,
                            "train_samples": len(train_indices),
                            "test_samples": len(test_indices),
                            "feature_count": int(mask.sum()),
                            "fit_seconds": fit_seconds,
                        }
                        row.update(classification_metrics(dataset.labels[test_indices], predicted))
                        metrics_rows.append(row)
                        prediction_rows.extend(
                            {
                                "sample_id": str(dataset.sample_ids[index]),
                                "person_id": str(dataset.person_ids[index]),
                                "label": int(dataset.labels[index]),
                                "prediction": int(prediction),
                                "model": model_name,
                                "feature_set": feature_set,
                                "variant": variant,
                                "seed": seed,
                                "fold": fold_index,
                            }
                            for index, prediction in zip(test_indices, predicted, strict=True)
                        )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    fold_assignments = pd.concat(assignment_frames, ignore_index=True)
    summary = metrics.groupby(["model", "feature_set", "variant"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in METRIC_COLUMNS},
        **{f"{metric}_std": (metric, "std") for metric in METRIC_COLUMNS},
        evaluations=("fold", "count"),
        seeds=("seed", "nunique"),
        folds_per_seed=("fold", "nunique"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )
    return EvaluationResult(metrics, predictions, fold_assignments, summary)


def _runtime_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-image", "scikit-learn", "scipy"]
    return {package: importlib.metadata.version(package) for package in packages}


def run_classical_evaluation(
    dataset_path: Path,
    feature_dir: Path,
    output_dir: Path,
    model_names: tuple[str, ...],
    feature_sets: tuple[str, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
    allow_unverified_identity: bool = False,
    allow_unverified_labels: bool = False,
) -> EvaluationResult:
    dataset = IrisDataset.load(dataset_path)
    require_identity_basis(dataset, allow_unverified=allow_unverified_identity)
    require_clinical_label_basis(dataset, allow_unverified=allow_unverified_labels)
    tables = {
        variant: FeatureTable.load(feature_dir / f"features_{variant}.npz") for variant in variants
    }
    result = evaluate_feature_tables(
        dataset,
        tables,
        model_names=model_names,
        feature_sets=feature_sets,
        seeds=seeds,
        n_splits=n_splits,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    result.predictions.to_csv(output_dir / "predictions.csv", index=False)
    result.fold_assignments.to_csv(output_dir / "fold_assignments.csv", index=False)
    result.summary.to_csv(output_dir / "summary.csv", index=False)
    config = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _runtime_versions(),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_metadata": dataset.metadata,
        "feature_files": {
            variant: {
                "path": str((feature_dir / f"features_{variant}.npz").resolve()),
                "sha256": sha256_file(feature_dir / f"features_{variant}.npz"),
            }
            for variant in variants
        },
        "models": list(model_names),
        "feature_sets": list(feature_sets),
        "variants": list(variants),
        "seeds": list(seeds),
        "n_splits": n_splits,
        "unverified_identity_explicitly_allowed": allow_unverified_identity,
        "unverified_labels_explicitly_allowed": allow_unverified_labels,
        "metrics": list(METRIC_COLUMNS),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
