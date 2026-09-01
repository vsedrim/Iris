from __future__ import annotations

import numpy as np

from iris_dm2.data import IrisDataset
from iris_dm2.evaluation import build_model, create_grouped_folds, evaluate_feature_tables
from iris_dm2.features import FeatureTable


def grouped_dataset() -> IrisDataset:
    person_ids = np.repeat([f"person-{index:02d}" for index in range(20)], 2)
    labels = np.repeat([0] * 10 + [1] * 10, 2).astype(np.uint8)
    images = np.zeros((40, 4, 4, 3), dtype=np.uint8)
    for index in range(len(images)):
        images[index] = index
    return IrisDataset(
        images=images,
        labels=labels,
        sample_ids=np.asarray([f"sample-{index:02d}" for index in range(40)]),
        person_ids=person_ids,
        eyes=np.tile(["L", "R"], 20),
    )


def feature_tables(dataset: IrisDataset) -> dict[str, FeatureTable]:
    person_signal = np.repeat(np.arange(20), 2).astype(np.float32)
    values = np.column_stack(
        [dataset.labels * 4 + person_signal / 100, person_signal, np.sin(person_signal)]
    ).astype(np.float32)
    common = {
        "names": np.asarray(["classic_signal", "morphology_index", "vascular_wave"]),
        "groups": np.asarray(["classic", "morphology", "vascular"]),
        "sample_ids": dataset.sample_ids,
    }
    return {
        "original": FeatureTable(values=values, variant="original", **common),
        "brightness_low": FeatureTable(
            values=values + np.asarray([0.05, 0, 0], dtype=np.float32),
            variant="brightness_low",
            **common,
        ),
    }


def test_grouped_folds_never_share_people() -> None:
    dataset = grouped_dataset()

    folds, assignments = create_grouped_folds(
        dataset.labels, dataset.person_ids, dataset.sample_ids, n_splits=5, seed=42
    )

    assert len(folds) == 5
    assert len(assignments) == len(dataset.images)
    for train_indices, test_indices in folds:
        assert not (set(dataset.person_ids[train_indices]) & set(dataset.person_ids[test_indices]))


def test_evaluation_reports_all_required_metrics_and_robustness_variant() -> None:
    dataset = grouped_dataset()

    result = evaluate_feature_tables(
        dataset,
        feature_tables(dataset),
        model_names=("logistic_regression",),
        feature_sets=("all", "classic"),
        seeds=(42,),
        n_splits=5,
    )

    assert len(result.metrics) == 20
    assert set(result.metrics["variant"]) == {"original", "brightness_low"}
    assert set(result.metrics["feature_set"]) == {"all", "classic"}
    for metric in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
        assert result.metrics[metric].between(0, 1).all()
        assert f"{metric}_mean" in result.summary
        assert f"{metric}_std" in result.summary
    assert len(result.predictions) == len(dataset.images) * 4


def test_mlp_does_not_create_sample_level_internal_validation_split() -> None:
    model = build_model("mlp", 42)

    assert model[-1].early_stopping is False
