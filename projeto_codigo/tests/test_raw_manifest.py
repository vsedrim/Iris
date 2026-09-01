from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from iris_dm2.data import PROVIDED_IDENTITY_STATUS, DatasetValidationError, IrisDataset
from iris_dm2.preprocessing import prepare_raw_manifest


def synthetic_eye(pupil_radius: int) -> np.ndarray:
    size = 256
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - 128) ** 2 + (y - 128) ** 2)
    image = np.full((size, size, 3), 235, dtype=np.uint8)
    image[distance <= 90] = [75, 125, 155]
    image[distance <= pupil_radius] = [8, 8, 8]
    return image


def base_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["control-left", "dm2-left"],
            "image_path": ["control.png", "dm2.png"],
            "person_id": ["person-control", "person-dm2"],
            "label": [0, 1],
            "eye": ["L", "L"],
            "diagnosis_source": ["clinical_record", "clinical_record"],
            "diagnosis_verified": [True, True],
        }
    )


def test_raw_manifest_requires_diagnosis_source(tmp_path: Path) -> None:
    manifest = base_manifest().drop(columns="diagnosis_source")
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="diagnosis_source"):
        prepare_raw_manifest(path, tmp_path / "output")


def test_raw_manifest_requires_explicit_diagnosis_verification(tmp_path: Path) -> None:
    manifest = base_manifest().drop(columns="diagnosis_verified")
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="diagnosis_verified"):
        prepare_raw_manifest(path, tmp_path / "output")


def test_raw_manifest_rejects_conflicting_labels_for_person(tmp_path: Path) -> None:
    manifest = base_manifest()
    manifest["person_id"] = "same-person"
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="conflicting labels"):
        prepare_raw_manifest(path, tmp_path / "output")


def test_raw_manifest_rejects_null_person_id(tmp_path: Path) -> None:
    manifest = base_manifest()
    manifest.loc[0, "person_id"] = None
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="cannot be null"):
        prepare_raw_manifest(path, tmp_path / "output")


def test_raw_manifest_prepares_geometry_and_records_provenance(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "control.png"), synthetic_eye(34)[..., ::-1])
    cv2.imwrite(str(tmp_path / "dm2.png"), synthetic_eye(42)[..., ::-1])
    manifest = base_manifest()
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    dataset = prepare_raw_manifest(path, tmp_path / "output")
    loaded = IrisDataset.load(tmp_path / "output" / "dataset.npz")

    assert dataset.images.shape == (2, 201, 720, 3)
    assert dataset.geometry.shape == (2, 19)
    assert dataset.metadata["person_identity_status"] == PROVIDED_IDENTITY_STATUS
    assert dataset.metadata["clinical_label_status"] == "verified_dm2_control"
    assert dataset.metadata["diagnosis_sources"] == ["clinical_record"]
    assert len(dataset.diagnosis_sources) == 2
    assert dataset.diagnosis_sources.tolist() == ["clinical_record", "clinical_record"]
    assert dataset.diagnosis_verified.tolist() == [True, True]
    assert all(len(value) == 64 for value in dataset.source_image_hashes)
    assert all(Path(value).is_absolute() for value in dataset.source_image_paths)
    assert len(dataset.metadata["source_manifest_sha256"]) == 64
    np.testing.assert_array_equal(loaded.diagnosis_sources, dataset.diagnosis_sources)
    np.testing.assert_array_equal(loaded.diagnosis_verified, dataset.diagnosis_verified)
    np.testing.assert_array_equal(loaded.source_image_paths, dataset.source_image_paths)
    np.testing.assert_array_equal(loaded.source_image_hashes, dataset.source_image_hashes)
    assert (tmp_path / "output" / "exclusions.json").exists()


def test_unverified_manifest_is_prepared_but_marked_unverified(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "control.png"), synthetic_eye(34)[..., ::-1])
    cv2.imwrite(str(tmp_path / "dm2.png"), synthetic_eye(42)[..., ::-1])
    manifest = base_manifest()
    manifest["diagnosis_verified"] = [True, False]
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    dataset = prepare_raw_manifest(path, tmp_path / "output")

    assert dataset.metadata["clinical_label_status"] == "unverified_diabetic_vs_control"


def test_raw_manifest_preserves_clinical_pupil_and_iris_measurements(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "control.png"), synthetic_eye(34)[..., ::-1])
    cv2.imwrite(str(tmp_path / "dm2.png"), synthetic_eye(42)[..., ::-1])
    manifest = base_manifest()
    manifest["pupil_diameter_mm"] = [3.1, 2.8]
    manifest["iris_thickness_mm"] = [0.41, 0.36]
    manifest["ocular_measurement_source"] = ["AS-OCT", "AS-OCT"]
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    dataset = prepare_raw_manifest(path, tmp_path / "output")

    assert dataset.geometry.shape == (2, 21)
    assert dataset.geometry_names[-2:].tolist() == [
        "pupil_diameter_mm",
        "iris_thickness_mm",
    ]
    np.testing.assert_allclose(dataset.geometry[:, -2:], [[3.1, 0.41], [2.8, 0.36]])
    assert dataset.metadata["ocular_measurement_sources"] == ["AS-OCT"]
    assert dataset.ocular_measurement_sources.tolist() == ["AS-OCT", "AS-OCT"]
    saved_manifest = pd.read_csv(tmp_path / "output" / "manifest.csv")
    assert saved_manifest["ocular_measurement_source"].tolist() == ["AS-OCT", "AS-OCT"]


def test_excluded_unverified_image_does_not_change_retained_label_status(
    tmp_path: Path,
) -> None:
    cv2.imwrite(str(tmp_path / "control.png"), synthetic_eye(34)[..., ::-1])
    cv2.imwrite(str(tmp_path / "dm2.png"), synthetic_eye(42)[..., ::-1])
    manifest = pd.concat(
        [
            base_manifest(),
            pd.DataFrame(
                {
                    "sample_id": ["excluded"],
                    "image_path": ["missing.png"],
                    "person_id": ["person-excluded"],
                    "label": [1],
                    "eye": ["R"],
                    "diagnosis_source": ["unverified_report"],
                    "diagnosis_verified": [False],
                }
            ),
        ],
        ignore_index=True,
    )
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    dataset = prepare_raw_manifest(path, tmp_path / "output")

    assert dataset.metadata["clinical_label_status"] == "verified_dm2_control"
    assert dataset.metadata["diagnosis_sources"] == ["clinical_record"]
    assert dataset.diagnosis_verified.tolist() == [True, True]


def test_raw_manifest_rejects_non_finite_clinical_measurement(tmp_path: Path) -> None:
    manifest = base_manifest()
    manifest["iris_thickness_mm"] = [0.41, np.inf]
    manifest["ocular_measurement_source"] = ["AS-OCT", "AS-OCT"]
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="positive measurements"):
        prepare_raw_manifest(path, tmp_path / "output")