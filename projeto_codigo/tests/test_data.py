from __future__ import annotations

import hashlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from iris_dm2.data import (
    DatasetValidationError,
    IrisDataset,
    inspect_pickle_globals,
    load_legacy_archive,
    require_clinical_label_basis,
    require_identity_basis,
    restricted_numpy_load,
)


def make_dataset() -> IrisDataset:
    images = np.zeros((4, 8, 12, 3), dtype=np.uint8)
    for index in range(len(images)):
        images[index, :, :, :] = index * 40
    return IrisDataset(
        images=images,
        labels=np.asarray([0, 0, 1, 1], dtype=np.uint8),
        sample_ids=np.asarray(["c-0", "c-1", "d-0", "d-1"]),
        person_ids=np.asarray(["pc-0", "pc-1", "pd-0", "pd-1"]),
        eyes=np.asarray(["L", "R", "L", "R"]),
        geometry=np.arange(8, dtype=np.float32).reshape(4, 2),
        geometry_names=np.asarray(["ratio", "offset"]),
        metadata={"source": "synthetic"},
    )


def test_safe_npz_round_trip_preserves_public_dataset_contract(tmp_path: Path) -> None:
    expected = make_dataset()

    expected.save(tmp_path)
    actual = IrisDataset.load(tmp_path / "dataset.npz")

    np.testing.assert_array_equal(actual.images, expected.images)
    np.testing.assert_array_equal(actual.labels, expected.labels)
    np.testing.assert_array_equal(actual.person_ids, expected.person_ids)
    np.testing.assert_array_equal(actual.geometry, expected.geometry)
    assert actual.metadata == expected.metadata


def test_sidecar_metadata_cannot_override_embedded_metadata(tmp_path: Path) -> None:
    dataset = make_dataset()
    dataset.metadata["person_identity_status"] = "unverified_source_row_proxy"
    dataset.save(tmp_path)
    dataset_hash_before = hashlib.sha256((tmp_path / "dataset.npz").read_bytes()).hexdigest()
    (tmp_path / "metadata.json").write_text(
        '{"person_identity_status":"provided_in_manifest"}', encoding="utf-8"
    )

    with pytest.raises(DatasetValidationError, match="does not match"):
        IrisDataset.load(tmp_path / "dataset.npz")

    dataset_hash_after = hashlib.sha256((tmp_path / "dataset.npz").read_bytes()).hexdigest()
    assert dataset_hash_after == dataset_hash_before


def test_restricted_loader_blocks_non_numpy_globals() -> None:
    payload = pickle.dumps(Path("not-allowed"), protocol=2)

    with pytest.raises(pickle.UnpicklingError, match="blocked global"):
        restricted_numpy_load(payload)


def test_pickle_inspector_resolves_protocol_four_stack_globals() -> None:
    payload = pickle.dumps(np.arange(4, dtype=np.uint8), protocol=4)

    references, parser_error = inspect_pickle_globals(payload)

    assert parser_error is None
    assert "None" not in references
    assert "<unresolved STACK_GLOBAL>" not in references
    assert any("numpy" in reference for reference in references)


def test_legacy_loader_deduplicates_and_converts_bgr_to_rgb(tmp_path: Path) -> None:
    control = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    control[0, :, :-1, :] = [10, 20, 30]
    control[1] = control[0]
    control[2, :, :-1, :] = [40, 50, 60]
    diabetic = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    diabetic[0, :, :-1, :] = [70, 80, 90]
    diabetic[1, :, :-1, :] = [100, 110, 120]
    archive_path = tmp_path / "Data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "Data/personBase_invert/controlImageArr.p", pickle.dumps(control, protocol=4)
        )
        archive.writestr(
            "Data/personBase_invert/diabeteImageArr.p", pickle.dumps(diabetic, protocol=4)
        )

    dataset = load_legacy_archive(archive_path, acknowledge_unverified_metadata=True)

    assert len(dataset.images) == 4
    assert dataset.metadata["exact_duplicates_removed"] == 1
    assert dataset.metadata["angular_seam_repaired"] is True
    np.testing.assert_array_equal(dataset.images[0, 0, 0], [30, 20, 10])
    np.testing.assert_array_equal(dataset.images[0, 0, -1], dataset.images[0, 0, 0])


def test_legacy_loader_rejects_layout_without_person_isolation(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="person-level isolation"):
        load_legacy_archive(tmp_path / "unused.zip", layout="all")


def test_legacy_loader_requires_explicit_unverified_metadata_acknowledgement(
    tmp_path: Path,
) -> None:
    with pytest.raises(DatasetValidationError, match="not verifiable"):
        load_legacy_archive(tmp_path / "unused.zip")


def test_legacy_loader_rejects_high_compression_ratio(tmp_path: Path) -> None:
    archive_path = tmp_path / "Data.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "Data/personBase_invert/controlImageArr.p", b"0" * (1024 * 1024)
        )
        archive.writestr("Data/personBase_invert/diabeteImageArr.p", b"unused")

    with pytest.raises(DatasetValidationError, match="compression ratio limit"):
        load_legacy_archive(archive_path, acknowledge_unverified_metadata=True)


@pytest.mark.parametrize("missing_id", ["", "nan", "None", "null"])
def test_dataset_rejects_missing_person_identifiers(missing_id: str) -> None:
    dataset = make_dataset()
    dataset.person_ids[0] = missing_id

    with pytest.raises(DatasetValidationError, match="missing identifiers"):
        dataset.validate()


def test_dataset_rejects_conflicting_labels_for_one_person() -> None:
    dataset = make_dataset()
    dataset.person_ids[2] = dataset.person_ids[0]

    with pytest.raises(DatasetValidationError, match="conflicting labels"):
        dataset.validate()


def test_unverified_identity_requires_explicit_evaluation_opt_in() -> None:
    dataset = make_dataset()
    dataset.metadata["person_identity_status"] = "unverified_source_row_proxy"

    with pytest.raises(DatasetValidationError, match="explicitly provisional"):
        require_identity_basis(dataset, allow_unverified=False)

    require_identity_basis(dataset, allow_unverified=True)


def test_unverified_clinical_labels_require_separate_opt_in() -> None:
    dataset = make_dataset()
    dataset.metadata["clinical_label_status"] = "unverified_diabetic_vs_control"

    with pytest.raises(DatasetValidationError, match="clinical label status"):
        require_clinical_label_basis(dataset, allow_unverified=False)

    require_clinical_label_basis(dataset, allow_unverified=True)


def test_dataset_rejects_clinical_status_that_conflicts_with_sample_verification() -> None:
    dataset = make_dataset()
    dataset.diagnosis_verified = np.ones(4, dtype=np.bool_)
    dataset.metadata["clinical_label_status"] = "unverified_diabetic_vs_control"

    with pytest.raises(DatasetValidationError, match="conflicts"):
        dataset.validate()


def test_verified_clinical_status_requires_per_sample_verification() -> None:
    dataset = make_dataset()
    dataset.metadata["clinical_label_status"] = "verified_dm2_control"

    with pytest.raises(DatasetValidationError, match="per-sample"):
        dataset.validate()

    with pytest.raises(DatasetValidationError, match="per-sample"):
        require_clinical_label_basis(dataset, allow_unverified=False)


def test_text_diagnosis_verification_cannot_be_coerced_to_true(tmp_path: Path) -> None:
    dataset = make_dataset()
    dataset.diagnosis_verified = np.ones(4, dtype=np.bool_)
    dataset.metadata["clinical_label_status"] = "verified_dm2_control"
    dataset.save(tmp_path)
    with np.load(tmp_path / "dataset.npz", allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["diagnosis_verified"] = np.asarray(["false"] * 4)
    np.savez_compressed(tmp_path / "dataset.npz", **values)

    with pytest.raises(DatasetValidationError, match="boolean dtype"):
        IrisDataset.load(tmp_path / "dataset.npz")
