from __future__ import annotations

import hashlib
import io
import json
import pickle
import pickletools
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PERSON_LEVEL_LAYOUTS = {"personBase", "personBase_invert"}
UNVERIFIED_IDENTITY_STATUS = "unverified_source_row_proxy"
PROVIDED_IDENTITY_STATUS = "provided_in_manifest"
VERIFIED_CLINICAL_LABEL_STATUS = "verified_dm2_control"
UNVERIFIED_CLINICAL_LABEL_STATUS = "unverified_diabetic_vs_control"
MAX_LEGACY_MEMBER_BYTES = 256 * 1024 * 1024
MAX_LEGACY_COMPRESSION_RATIO = 50.0
ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}
GLOBAL_OPCODE_PATTERN = re.compile(
    rb"c([A-Za-z_][A-Za-z0-9_.]*)\n([A-Za-z_][A-Za-z0-9_.]*)\n"
)


class DatasetValidationError(ValueError):
    pass


class RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"blocked global: {module}.{name}")
        return super().find_class(module, name)


def restricted_numpy_load(payload: bytes) -> np.ndarray:
    value = RestrictedNumpyUnpickler(io.BytesIO(payload), encoding="latin1").load()
    if not isinstance(value, np.ndarray):
        raise pickle.UnpicklingError(f"expected numpy.ndarray, received {type(value).__name__}")
    return value


def inspect_pickle_globals(payload: bytes) -> tuple[list[str], str | None]:
    references: set[str] = set()
    recent_strings: list[str] = []
    parser_error: str | None = None
    try:
        for opcode, argument, _ in pickletools.genops(payload):
            if opcode.name == "GLOBAL":
                references.add(str(argument))
            elif opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
                recent_strings.append(str(argument))
                recent_strings = recent_strings[-2:]
            elif opcode.name == "STACK_GLOBAL":
                if len(recent_strings) != 2:
                    references.add("<unresolved STACK_GLOBAL>")
                else:
                    references.add(f"{recent_strings[0]} {recent_strings[1]}")
    except (UnicodeDecodeError, ValueError) as error:
        parser_error = f"{type(error).__name__}: {error}"

    references.update(
        f"{module.decode('ascii')} {name.decode('ascii')}"
        for module, name in GLOBAL_OPCODE_PATTERN.findall(payload)
    )
    return sorted(references), parser_error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_metadata(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


@dataclass
class IrisDataset:
    images: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    person_ids: np.ndarray
    eyes: np.ndarray
    diagnosis_sources: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="<U1")
    )
    diagnosis_verified: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )
    source_image_paths: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="<U1")
    )
    source_image_hashes: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="<U1")
    )
    geometry: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    geometry_names: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="<U1"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise DatasetValidationError("images must have shape (samples, height, width, 3)")
        sample_count = self.images.shape[0]
        arrays = {
            "labels": self.labels,
            "sample_ids": self.sample_ids,
            "person_ids": self.person_ids,
            "eyes": self.eyes,
        }
        for name, value in arrays.items():
            if value.ndim != 1 or len(value) != sample_count:
                raise DatasetValidationError(
                    f"{name} must be one-dimensional with {sample_count} entries"
                )
        for name, value in {
            "diagnosis_sources": self.diagnosis_sources,
            "diagnosis_verified": self.diagnosis_verified,
            "source_image_paths": self.source_image_paths,
            "source_image_hashes": self.source_image_hashes,
        }.items():
            if value.size and (value.ndim != 1 or len(value) != sample_count):
                raise DatasetValidationError(
                    f"{name} must be empty or one-dimensional with {sample_count} entries"
                )
            if value.size and any(
                str(item).strip().lower() in {"", "nan", "none", "null", "<na>"}
                for item in value
            ):
                raise DatasetValidationError(f"{name} cannot contain missing values")
        if self.diagnosis_verified.size and self.diagnosis_verified.dtype != np.bool_:
            raise DatasetValidationError("diagnosis_verified must use boolean dtype")
        if self.source_image_hashes.size and any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in self.source_image_hashes
        ):
            raise DatasetValidationError("source_image_hashes must contain SHA-256 hex digests")
        if self.metadata.get("person_identity_status") == PROVIDED_IDENTITY_STATUS:
            missing_provenance = [
                name
                for name, values in {
                    "diagnosis_sources": self.diagnosis_sources,
                    "diagnosis_verified": self.diagnosis_verified,
                    "source_image_paths": self.source_image_paths,
                    "source_image_hashes": self.source_image_hashes,
                }.items()
                if len(values) != sample_count
            ]
            if missing_provenance:
                raise DatasetValidationError(
                    f"manifest-based dataset lacks provenance: {missing_provenance}"
                )
        if self.diagnosis_verified.size:
            expected_status = (
                VERIFIED_CLINICAL_LABEL_STATUS
                if self.diagnosis_verified.all()
                else UNVERIFIED_CLINICAL_LABEL_STATUS
            )
            actual_status = self.metadata.get("clinical_label_status", "unknown")
            if actual_status != expected_status:
                raise DatasetValidationError(
                    f"clinical label status {actual_status!r} conflicts with per-sample "
                    f"verification values ({expected_status!r})"
                )
        elif self.metadata.get("clinical_label_status") == VERIFIED_CLINICAL_LABEL_STATUS:
            raise DatasetValidationError(
                "verified clinical label status requires per-sample diagnosis verification"
            )
        if set(np.unique(self.labels).tolist()) != {0, 1}:
            raise DatasetValidationError("labels must contain both binary classes 0 and 1")
        if len(np.unique(self.sample_ids)) != sample_count:
            raise DatasetValidationError("sample_ids must be unique")
        for name, values in {
            "sample_ids": self.sample_ids,
            "person_ids": self.person_ids,
        }.items():
            normalized = {str(value).strip().lower() for value in values}
            invalid = normalized & {"", "nan", "none", "null", "<na>"}
            if invalid:
                raise DatasetValidationError(f"{name} cannot contain missing identifiers")
        for person_id in np.unique(self.person_ids):
            person_labels = np.unique(self.labels[self.person_ids == person_id])
            if len(person_labels) != 1:
                raise DatasetValidationError(
                    f"person {person_id!r} has conflicting labels: {person_labels.tolist()}"
                )
        if self.geometry.size:
            if self.geometry.shape[0] != sample_count:
                raise DatasetValidationError("geometry row count must match image count")
            if self.geometry.shape[1] != len(self.geometry_names):
                raise DatasetValidationError("geometry_names must match geometry columns")
        samples_by_hash: dict[bytes, list[str]] = {}
        for sample_id, image in zip(self.sample_ids, self.images, strict=True):
            digest = hashlib.sha256(image.tobytes()).digest()
            samples_by_hash.setdefault(digest, []).append(str(sample_id))
        duplicate_groups = [ids for ids in samples_by_hash.values() if len(ids) > 1]
        if duplicate_groups:
            raise DatasetValidationError(
                "exact duplicate images would create leakage: "
                + "; ".join(", ".join(ids) for ids in duplicate_groups)
            )

    def save(self, output_dir: Path) -> None:
        self.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_json = canonical_metadata(self.metadata)
        np.savez_compressed(
            output_dir / "dataset.npz",
            images=self.images,
            labels=self.labels,
            sample_ids=self.sample_ids,
            person_ids=self.person_ids,
            eyes=self.eyes,
            diagnosis_sources=self.diagnosis_sources,
            diagnosis_verified=self.diagnosis_verified,
            source_image_paths=self.source_image_paths,
            source_image_hashes=self.source_image_hashes,
            geometry=self.geometry,
            geometry_names=self.geometry_names,
            metadata_json=np.asarray(metadata_json),
        )
        positive_label_name = self.metadata.get("positive_label_name", "dm2")
        negative_label_name = self.metadata.get("negative_label_name", "control")
        manifest_data: dict[str, np.ndarray] = {
            "sample_id": self.sample_ids,
            "person_id": self.person_ids,
            "label": self.labels,
            "label_name": np.where(
                self.labels == 1, positive_label_name, negative_label_name
            ),
            "eye": self.eyes,
        }
        for name, values in {
            "diagnosis_source": self.diagnosis_sources,
            "diagnosis_verified": self.diagnosis_verified,
            "source_image_path": self.source_image_paths,
            "source_image_sha256": self.source_image_hashes,
        }.items():
            if values.size:
                manifest_data[name] = values
        pd.DataFrame(manifest_data).to_csv(output_dir / "manifest.csv", index=False)
        (output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, dataset_path: Path) -> IrisDataset:
        with np.load(dataset_path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise DatasetValidationError("dataset lacks embedded canonical metadata")

            def optional_array(name: str) -> np.ndarray:
                return archive[name] if name in archive.files else np.empty(0, dtype="<U1")

            embedded_metadata_json = str(archive["metadata_json"].item())
            dataset = cls(
                images=archive["images"],
                labels=archive["labels"],
                sample_ids=archive["sample_ids"],
                person_ids=archive["person_ids"],
                eyes=archive["eyes"],
                diagnosis_sources=optional_array("diagnosis_sources"),
                diagnosis_verified=optional_array("diagnosis_verified"),
                source_image_paths=optional_array("source_image_paths"),
                source_image_hashes=optional_array("source_image_hashes"),
                geometry=archive["geometry"],
                geometry_names=archive["geometry_names"],
                metadata=json.loads(embedded_metadata_json),
            )
        metadata_path = dataset_path.with_name("metadata.json")
        if metadata_path.exists():
            sidecar_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if canonical_metadata(sidecar_metadata) != embedded_metadata_json:
                raise DatasetValidationError(
                    "metadata.json does not match metadata embedded in dataset.npz"
                )
        dataset.validate()
        return dataset


def _repair_legacy_angular_seam(images: np.ndarray) -> tuple[np.ndarray, bool]:
    seam_is_empty = bool(np.all(images[:, :, -1, :] == 0))
    preceding_column_has_data = bool(np.any(images[:, :, -2, :] != 0))
    if seam_is_empty and preceding_column_has_data:
        images = images.copy()
        images[:, :, -1, :] = images[:, :, 0, :]
        return images, True
    return images, False


def _drop_exact_duplicates(
    images: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    first_index_by_hash: dict[bytes, int] = {}
    duplicate_ids_by_hash: dict[bytes, list[str]] = {}
    retained_indices: list[int] = []

    for index, (sample_id, image) in enumerate(zip(sample_ids, images, strict=True)):
        digest = hashlib.sha256(image.tobytes()).digest()
        if digest not in first_index_by_hash:
            first_index_by_hash[digest] = index
            retained_indices.append(index)
            duplicate_ids_by_hash[digest] = [str(sample_id)]
        else:
            first_index = first_index_by_hash[digest]
            if labels[first_index] != labels[index]:
                conflict = f"{sample_ids[first_index]}, {sample_id}"
                raise DatasetValidationError(f"identical image has conflicting labels: {conflict}")
            duplicate_ids_by_hash[digest].append(str(sample_id))

    duplicate_groups = [
        {"retained": ids[0], "removed": ids[1:]}
        for ids in duplicate_ids_by_hash.values()
        if len(ids) > 1
    ]
    retained = np.asarray(retained_indices, dtype=np.int64)
    return images[retained], labels[retained], sample_ids[retained], duplicate_groups


def read_legacy_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.flag_bits & 0x1:
        raise DatasetValidationError(f"encrypted legacy member is not supported: {name}")
    if info.file_size > MAX_LEGACY_MEMBER_BYTES:
        raise DatasetValidationError(f"legacy member exceeds size limit: {name}")
    compression_ratio = info.file_size / max(info.compress_size, 1)
    if compression_ratio > MAX_LEGACY_COMPRESSION_RATIO:
        raise DatasetValidationError(f"legacy member exceeds compression ratio limit: {name}")
    payload = archive.read(info)
    if len(payload) != info.file_size:
        raise DatasetValidationError(f"legacy member size mismatch: {name}")
    return payload


def load_legacy_archive(
    archive_path: Path,
    layout: str = "personBase_invert",
    acknowledge_unverified_metadata: bool = False,
) -> IrisDataset:
    if layout not in PERSON_LEVEL_LAYOUTS:
        raise DatasetValidationError(
            f"layout {layout!r} cannot prove person-level isolation; "
            f"choose one of {sorted(PERSON_LEVEL_LAYOUTS)}"
        )
    if not acknowledge_unverified_metadata:
        raise DatasetValidationError(
            "legacy subject identity and DM2 subtype are not verifiable; pass "
            "acknowledge_unverified_metadata=True only for explicitly provisional analysis"
        )

    prefix = f"Data/{layout}"
    with zipfile.ZipFile(archive_path) as archive:
        control = restricted_numpy_load(
            read_legacy_member(archive, f"{prefix}/controlImageArr.p")
        )
        diabetic = restricted_numpy_load(
            read_legacy_member(archive, f"{prefix}/diabeteImageArr.p")
        )

    if control.ndim != 4 or diabetic.ndim != 4 or control.shape[1:] != diabetic.shape[1:]:
        raise DatasetValidationError("legacy control and diabetic image tensors are incompatible")
    if control.dtype != np.uint8 or diabetic.dtype != np.uint8:
        raise DatasetValidationError("legacy images must use uint8 pixels")

    images = np.concatenate([control, diabetic], axis=0)
    images, seam_repaired = _repair_legacy_angular_seam(images)
    images = images[..., ::-1].copy()
    labels = np.concatenate(
        [np.zeros(len(control), dtype=np.uint8), np.ones(len(diabetic), dtype=np.uint8)]
    )
    source_sample_ids = np.asarray(
        [f"control-{index:03d}" for index in range(len(control))]
        + [f"reported-diabetic-{index:03d}" for index in range(len(diabetic))]
    )
    images, labels, sample_ids, duplicate_groups = _drop_exact_duplicates(
        images, labels, source_sample_ids
    )
    effective_control_count = int((labels == 0).sum())
    effective_dm2_count = int((labels == 1).sum())

    dataset = IrisDataset(
        images=images,
        labels=labels,
        sample_ids=sample_ids,
        person_ids=sample_ids.copy(),
        eyes=np.full(len(images), "unknown"),
        diagnosis_sources=np.full(len(images), "legacy_archive_unverified"),
        diagnosis_verified=np.zeros(len(images), dtype=np.bool_),
        source_image_paths=np.asarray(
            [
                f"{prefix}/controlImageArr.p[{sample_id.rsplit('-', 1)[1]}]"
                if sample_id.startswith("control-")
                else f"{prefix}/diabeteImageArr.p[{sample_id.rsplit('-', 1)[1]}]"
                for sample_id in sample_ids
            ]
        ),
        source_image_hashes=np.asarray(
            [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
        ),
        geometry=np.empty((len(images), 0), dtype=np.float32),
        metadata={
            "archive_sha256": sha256_file(archive_path),
            "source_archive": archive_path.name,
            "source_layout": layout,
            "source_publication_status": "retracted",
            "source_publication_doi": "10.1109/ICBME.2018.8703564",
            "retraction_notice_doi": "10.1109/ICBME45317.2018.10207763",
            "sample_count": int(len(images)),
            "source_control_count": int(len(control)),
            "source_reported_diabetic_count": int(len(diabetic)),
            "effective_control_count": effective_control_count,
            "effective_reported_diabetic_count": effective_dm2_count,
            "exact_duplicates_removed": int(
                sum(len(group["removed"]) for group in duplicate_groups)
            ),
            "duplicate_groups": duplicate_groups,
            "color_conversion": "legacy OpenCV BGR to RGB",
            "angular_seam_repaired": seam_repaired,
            "person_identity_status": UNVERIFIED_IDENTITY_STATUS,
            "clinical_label_status": UNVERIFIED_CLINICAL_LABEL_STATUS,
            "positive_label_name": "reported_diabetic_unverified",
            "negative_label_name": "reported_control_unverified",
            "identity_basis": (
                "The personBase layout row count matches the reported subject count. Synthetic "
                "row IDs are not proof of subject identity."
            ),
            "geometry_available": False,
            "limitations": [
                "The archive contains normalized strips without raw images or contours.",
                "Pupil diameter and boundary morphology cannot be recovered from this archive.",
                "The positive class is reported diabetic; DM2 subtype is not verifiable.",
                "The publication associated with this archive has been retracted by IEEE.",
            ],
        },
    )
    dataset.validate()
    return dataset


def prepare_legacy_archive(
    archive_path: Path,
    output_dir: Path,
    layout: str,
    acknowledge_unverified_metadata: bool = False,
) -> IrisDataset:
    dataset = load_legacy_archive(
        archive_path,
        layout=layout,
        acknowledge_unverified_metadata=acknowledge_unverified_metadata,
    )
    dataset.save(output_dir)
    return dataset


def require_identity_basis(dataset: IrisDataset, allow_unverified: bool) -> None:
    status = dataset.metadata.get("person_identity_status", "unknown")
    if status == PROVIDED_IDENTITY_STATUS:
        return
    if not allow_unverified:
        raise DatasetValidationError(
            f"group identity status is {status!r}; pass allow_unverified=True only for an "
            "explicitly provisional analysis"
        )


def require_clinical_label_basis(dataset: IrisDataset, allow_unverified: bool) -> None:
    status = dataset.metadata.get("clinical_label_status", "unknown")
    if status == VERIFIED_CLINICAL_LABEL_STATUS:
        if len(dataset.diagnosis_verified) != len(dataset.labels) or not bool(
            dataset.diagnosis_verified.all()
        ):
            raise DatasetValidationError(
                "verified clinical labels require complete per-sample verification"
            )
        return
    if not allow_unverified:
        raise DatasetValidationError(
            f"clinical label status is {status!r}; pass allow_unverified=True only for an "
            "explicitly provisional analysis"
        )
