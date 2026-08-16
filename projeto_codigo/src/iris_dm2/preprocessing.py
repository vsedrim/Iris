from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from iris_dm2.data import (
    PROVIDED_IDENTITY_STATUS,
    UNVERIFIED_CLINICAL_LABEL_STATUS,
    VERIFIED_CLINICAL_LABEL_STATUS,
    DatasetValidationError,
    IrisDataset,
    sha256_file,
)

MIN_SEGMENTATION_QUALITY = 0.30
MIN_ANNULUS_WIDTH_PIXELS = 3.0

GEOMETRY_FEATURE_NAMES = np.asarray(
    [
        "pupil_diameter_px",
        "iris_diameter_px",
        "pupil_iris_diameter_ratio",
        "pupil_iris_area_ratio",
        "center_offset_normalized",
        "pupil_circularity",
        "iris_circularity",
        "pupil_radial_std",
        "pupil_radial_max_abs",
        "pupil_border_high_frequency_energy",
        "pupil_border_lobe_count",
        "iris_radial_std",
        "iris_radial_max_abs",
        "iris_border_high_frequency_energy",
        "iris_border_lobe_count",
        "segmentation_quality",
    ]
)


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class SegmentationResult:
    pupil: Circle
    iris: Circle
    pupil_contour: np.ndarray
    iris_contour: np.ndarray
    quality: float


def _circle_contour(circle: Circle, point_count: int = 720) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, point_count, endpoint=False)
    points = np.column_stack(
        [circle.x + circle.radius * np.cos(angles), circle.y + circle.radius * np.sin(angles)]
    )
    return points.astype(np.float32).reshape(-1, 1, 2)


def _contour_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour.astype(np.float32))
    perimeter = cv2.arcLength(contour.astype(np.float32), True)
    if area <= 0 or perimeter <= 0:
        return 0.0
    return float(np.clip(4 * np.pi * area / (perimeter * perimeter), 0.0, 1.0))


def _detect_pupil(gray: np.ndarray) -> tuple[Circle, np.ndarray, float]:
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_center = np.asarray([width / 2, height / 2], dtype=np.float32)
    max_radius = min(height, width) * 0.28
    candidates: list[tuple[float, Circle, np.ndarray, float]] = []
    thresholds = {float(np.percentile(blurred, percentile)) for percentile in (3, 5, 10, 15, 20)}
    thresholds.add(min(float(otsu_threshold), float(np.percentile(blurred, 30))))
    for threshold in sorted(thresholds):
        mask = np.where(blurred <= threshold, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 4 or radius > max_radius:
                continue
            center_distance = np.linalg.norm(np.asarray([x, y]) - image_center) / min(height, width)
            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            pupil_intensity = float(cv2.mean(gray, mask=contour_mask)[0])
            outer_mask = np.zeros_like(gray)
            cv2.circle(
                outer_mask,
                (int(round(x)), int(round(y))),
                int(round(radius * 1.35)),
                255,
                thickness=cv2.FILLED,
            )
            ring_mask = cv2.subtract(outer_mask, contour_mask)
            surrounding_intensity = float(cv2.mean(gray, mask=ring_mask)[0])
            contrast_quality = float(
                np.clip((surrounding_intensity - pupil_intensity) / 80.0, 0.0, 1.0)
            )
            centrality_quality = float(np.clip(1.0 - center_distance / 0.35, 0.0, 1.0))
            quality = 0.75 * contrast_quality + 0.25 * centrality_quality
            darkness = pupil_intensity / 255.0
            score = darkness + 0.8 * center_distance
            candidates.append((score, Circle(x, y, radius), contour, quality))

    if not candidates:
        raise DatasetValidationError("pupil segmentation found no plausible dark circular region")
    _, circle, contour, quality = min(candidates, key=lambda item: item[0])
    return circle, contour.astype(np.float32), quality


def _fit_circle(points: np.ndarray) -> Circle:
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    design = np.column_stack([2 * x, 2 * y, np.ones(len(points))])
    target = x**2 + y**2
    center_x, center_y, constant = np.linalg.lstsq(design, target, rcond=None)[0]
    radius_squared = constant + center_x**2 + center_y**2
    if radius_squared <= 0:
        raise DatasetValidationError("circle fit produced a non-positive radius")
    return Circle(float(center_x), float(center_y), float(np.sqrt(radius_squared)))


def _sample_gradient_ring(
    gradient: np.ndarray,
    center_x: float,
    center_y: float,
    radii: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    height, width = gradient.shape
    x = np.rint(center_x + radii[:, None] * np.cos(angles)).astype(int)
    y = np.rint(center_y + radii[:, None] * np.sin(angles)).astype(int)
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    return gradient[y, x]


def _closed_iris_boundary(
    gray: np.ndarray,
    gradient: np.ndarray,
    pupil: Circle,
    minimum_radius: float,
    maximum_radius: float,
) -> tuple[Circle, np.ndarray, float] | None:
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    median = float(np.median(blurred))
    edges = cv2.Canny(
        blurred,
        threshold1=max(10, int(0.5 * median)),
        threshold2=max(30, int(1.2 * median)),
    )
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates: list[tuple[float, Circle, np.ndarray, float]] = []
    strong_gradient = float(np.quantile(gradient, 0.95)) + 1e-6
    for contour in contours:
        if len(contour) < 40:
            continue
        if cv2.pointPolygonTest(contour, (pupil.x, pupil.y), False) < 0:
            continue
        points = contour.reshape(-1, 2).astype(np.float32)
        circle = _fit_circle(points)
        if not minimum_radius <= circle.radius <= maximum_radius:
            continue
        if np.hypot(circle.x - pupil.x, circle.y - pupil.y) > pupil.radius * 1.5:
            continue
        height, width = gray.shape
        coordinates = np.rint(points).astype(int)
        x = np.clip(coordinates[:, 0], 0, width - 1)
        y = np.clip(coordinates[:, 1], 0, height - 1)
        edge_support = float(np.mean(gradient[y, x]))
        edge_quality = float(np.clip(edge_support / strong_gradient, 0.0, 1.0))
        score = edge_support + 0.2 * circle.radius
        candidates.append((score, circle, contour.astype(np.float32), edge_quality))
    if not candidates:
        return None
    _, circle, contour, edge_quality = max(candidates, key=lambda item: item[0])
    return circle, contour, edge_quality


def _detect_iris(gray: np.ndarray, pupil: Circle) -> tuple[Circle, np.ndarray, float]:
    height, width = gray.shape
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    minimum_radius = max(pupil.radius * 1.45, pupil.radius + 5)
    pupil_border_limit = min(pupil.x, width - pupil.x, pupil.y, height - pupil.y)
    global_maximum_radius = min(pupil_border_limit * 0.98, pupil.radius * 5.0)
    closed_boundary = _closed_iris_boundary(
        gray,
        gradient,
        pupil,
        minimum_radius,
        global_maximum_radius,
    )
    if closed_boundary is not None:
        return closed_boundary

    search_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    maximum_offset = max(3.0, pupil.radius * 0.30)
    offsets = np.linspace(-maximum_offset, maximum_offset, 5)
    candidates: list[tuple[float, float, float, float]] = []
    all_scores: list[float] = []

    for offset_x in offsets:
        for offset_y in offsets:
            center_x = pupil.x + offset_x
            center_y = pupil.y + offset_y
            border_limit = min(center_x, width - center_x, center_y, height - center_y)
            maximum_radius = min(border_limit * 0.98, pupil.radius * 5.0)
            if maximum_radius <= minimum_radius:
                continue
            radii = np.arange(
                int(np.ceil(minimum_radius)), int(np.floor(maximum_radius)) + 1, 2
            )
            samples = _sample_gradient_ring(
                gradient, center_x, center_y, radii, search_angles
            )
            scores = np.quantile(samples, 0.90, axis=1)
            all_scores.extend(scores.tolist())
            best_index = int(np.argmax(scores))
            center_penalty = 0.02 * np.hypot(offset_x, offset_y) / pupil.radius
            candidates.append(
                (
                    float(scores[best_index] - center_penalty),
                    center_x,
                    center_y,
                    float(radii[best_index]),
                )
            )

    if not candidates:
        raise DatasetValidationError("iris radius search range is empty")
    best_score, center_x, center_y, approximate_radius = max(candidates)
    boundary_angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    radius_window = max(5, int(round(approximate_radius * 0.10)))
    boundary_radii = np.arange(
        max(int(minimum_radius), int(round(approximate_radius)) - radius_window),
        int(round(approximate_radius)) + radius_window + 1,
    )
    boundary_samples = _sample_gradient_ring(
        gradient, center_x, center_y, boundary_radii, boundary_angles
    )
    selected_radii = boundary_radii[np.argmax(boundary_samples, axis=0)].astype(
        np.float32
    )
    selected_radii = cv2.GaussianBlur(
        np.concatenate([selected_radii[-8:], selected_radii, selected_radii[:8]])[None, :],
        (17, 1),
        0,
    )[0, 8:-8]
    points = np.column_stack(
        [
            center_x + selected_radii * np.cos(boundary_angles),
            center_y + selected_radii * np.sin(boundary_angles),
        ]
    ).astype(np.float32)
    fitted_circle = _fit_circle(points)
    score_array = np.asarray(all_scores, dtype=np.float32)
    quality = best_score / (float(score_array.mean() + score_array.std()) + 1e-6)
    return (
        fitted_circle,
        points.reshape(-1, 1, 2),
        float(np.clip(quality / 2.0, 0.0, 1.0)),
    )


def segment_iris(image_rgb: np.ndarray) -> SegmentationResult:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise DatasetValidationError("raw image must be an RGB uint8 array")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    pupil, pupil_contour, pupil_quality = _detect_pupil(gray)
    iris, iris_contour, iris_quality = _detect_iris(gray, pupil)
    if iris.radius <= pupil.radius:
        raise DatasetValidationError("segmented iris must be larger than pupil")
    quality = float(np.clip((pupil_quality + iris_quality) / 2.0, 0.0, 1.0))
    segmentation = SegmentationResult(
        pupil=pupil,
        iris=iris,
        pupil_contour=pupil_contour,
        iris_contour=iris_contour,
        quality=quality,
    )
    validate_segmentation(segmentation, image_rgb.shape[:2])
    return segmentation


def rubber_sheet_normalize(
    image_rgb: np.ndarray,
    segmentation: SegmentationResult,
    output_shape: tuple[int, int] = (201, 720),
) -> np.ndarray:
    validate_segmentation(segmentation, image_rgb.shape[:2])
    radial_count, angular_count = output_shape
    angles = np.linspace(0, 2 * np.pi, angular_count, endpoint=False, dtype=np.float32)
    radial = np.linspace(0, 1, radial_count, dtype=np.float32)[:, None]
    pupil_radii = _contour_radius_profile(
        segmentation.pupil_contour, segmentation.pupil, angles
    )
    iris_radii = _contour_radius_profile(
        segmentation.iris_contour, segmentation.iris, angles
    )
    pupil_x = segmentation.pupil.x + pupil_radii * np.cos(angles)
    pupil_y = segmentation.pupil.y + pupil_radii * np.sin(angles)
    iris_x = segmentation.iris.x + iris_radii * np.cos(angles)
    iris_y = segmentation.iris.y + iris_radii * np.sin(angles)
    map_x = (1.0 - radial) * pupil_x + radial * iris_x
    map_y = (1.0 - radial) * pupil_y + radial * iris_y
    return cv2.remap(
        image_rgb,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def validate_segmentation(
    segmentation: SegmentationResult, image_shape: tuple[int, int]
) -> None:
    if segmentation.quality < MIN_SEGMENTATION_QUALITY:
        raise DatasetValidationError(
            f"segmentation quality {segmentation.quality:.3f} is below "
            f"{MIN_SEGMENTATION_QUALITY:.3f}"
        )
    if segmentation.pupil.radius <= 0 or segmentation.iris.radius <= segmentation.pupil.radius:
        raise DatasetValidationError("segmentation radii do not define an iris annulus")

    iris_contour = segmentation.iris_contour.astype(np.float32)
    pupil_points = segmentation.pupil_contour.reshape(-1, 2)
    outside_count = sum(
        cv2.pointPolygonTest(iris_contour, tuple(map(float, point)), False) < 0
        for point in pupil_points
    )
    if outside_count:
        raise DatasetValidationError(
            f"pupil contour crosses iris boundary at {outside_count} points"
        )

    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False, dtype=np.float32)
    pupil_radii = _contour_radius_profile(
        segmentation.pupil_contour, segmentation.pupil, angles
    )
    iris_radii = _contour_radius_profile(
        segmentation.iris_contour, segmentation.iris, angles
    )
    pupil_points_angular = np.column_stack(
        [
            segmentation.pupil.x + pupil_radii * np.cos(angles),
            segmentation.pupil.y + pupil_radii * np.sin(angles),
        ]
    )
    iris_points_angular = np.column_stack(
        [
            segmentation.iris.x + iris_radii * np.cos(angles),
            segmentation.iris.y + iris_radii * np.sin(angles),
        ]
    )
    annulus_width = np.linalg.norm(iris_points_angular - pupil_points_angular, axis=1)
    minimum_width = max(MIN_ANNULUS_WIDTH_PIXELS, segmentation.pupil.radius * 0.10)
    if float(np.min(annulus_width)) < minimum_width:
        raise DatasetValidationError(
            f"iris annulus is thinner than {minimum_width:.2f} pixels"
        )

    height, width = image_shape
    all_points = np.vstack([pupil_points_angular, iris_points_angular])
    if (
        np.any(all_points[:, 0] < 0)
        or np.any(all_points[:, 0] > width - 1)
        or np.any(all_points[:, 1] < 0)
        or np.any(all_points[:, 1] > height - 1)
    ):
        raise DatasetValidationError("segmentation contour leaves the image bounds")


def _contour_radius_profile(
    contour: np.ndarray, circle: Circle, target_angles: np.ndarray
) -> np.ndarray:
    contour_points = contour.reshape(-1, 2).astype(np.float64)
    centered = contour_points - np.asarray([circle.x, circle.y])
    contour_angles = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2 * np.pi)
    contour_radii = np.linalg.norm(centered, axis=1)
    order = np.argsort(contour_angles)
    sorted_angles = contour_angles[order]
    sorted_radii = contour_radii[order]
    unique_angles, unique_indices = np.unique(sorted_angles, return_index=True)
    unique_radii = sorted_radii[unique_indices]
    periodic_angles = np.concatenate(
        [unique_angles[-1:] - 2 * np.pi, unique_angles, unique_angles[:1] + 2 * np.pi]
    )
    periodic_radii = np.concatenate(
        [unique_radii[-1:], unique_radii, unique_radii[:1]]
    )
    return np.interp(target_angles, periodic_angles, periodic_radii).astype(np.float32)


def _radial_deviation_features(contour: np.ndarray, circle: Circle) -> tuple[float, ...]:
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    distances = _contour_radius_profile(contour, circle, angles)
    radial_deviation = (distances - circle.radius) / circle.radius
    centered = radial_deviation - radial_deviation.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    high_frequency_energy = float(spectrum[3:].sum() / (spectrum.sum() + 1e-12))
    peaks, _ = find_peaks(np.abs(centered), prominence=max(float(np.std(centered)), 1e-4))
    return (
        float(np.std(radial_deviation)),
        float(np.max(np.abs(radial_deviation))),
        high_frequency_energy,
        float(len(peaks)),
    )


def geometric_features(segmentation: SegmentationResult) -> np.ndarray:
    pupil_deviation = _radial_deviation_features(
        segmentation.pupil_contour, segmentation.pupil
    )
    iris_deviation = _radial_deviation_features(
        segmentation.iris_contour, segmentation.iris
    )
    center_offset = np.hypot(
        segmentation.pupil.x - segmentation.iris.x,
        segmentation.pupil.y - segmentation.iris.y,
    )
    diameter_ratio = segmentation.pupil.radius / segmentation.iris.radius
    pupil_area = cv2.contourArea(segmentation.pupil_contour.astype(np.float32))
    iris_area = cv2.contourArea(segmentation.iris_contour.astype(np.float32))
    if pupil_area <= 0 or iris_area <= pupil_area:
        raise DatasetValidationError("segmentation contours do not define valid annular areas")
    area_ratio = pupil_area / iris_area
    return np.asarray(
        [
            2 * segmentation.pupil.radius,
            2 * segmentation.iris.radius,
            diameter_ratio,
            area_ratio,
            center_offset / segmentation.iris.radius,
            _contour_circularity(segmentation.pupil_contour),
            _contour_circularity(segmentation.iris_contour),
            *pupil_deviation,
            *iris_deviation,
            segmentation.quality,
        ],
        dtype=np.float32,
    )


def prepare_raw_manifest(manifest_path: Path, output_dir: Path) -> IrisDataset:
    manifest = pd.read_csv(manifest_path)
    required_columns = {
        "image_path",
        "person_id",
        "label",
        "eye",
        "diagnosis_source",
        "diagnosis_verified",
    }
    missing = required_columns - set(manifest.columns)
    if missing:
        raise DatasetValidationError(f"raw manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise DatasetValidationError("raw manifest cannot be empty")
    if manifest[list(required_columns)].isna().any().any():
        raise DatasetValidationError("raw manifest required fields cannot be null")

    for column in ("image_path", "person_id", "eye", "diagnosis_source"):
        manifest[column] = manifest[column].astype(str).str.strip()
        if manifest[column].str.lower().isin({"", "nan", "none", "null", "<na>"}).any():
            raise DatasetValidationError(f"raw manifest column {column!r} has missing values")
    manifest["label"] = pd.to_numeric(manifest["label"], errors="raise")
    if not manifest["label"].isin([0, 1]).all():
        raise DatasetValidationError("raw manifest labels must be 0 (control) or 1 (DM2)")
    manifest["label"] = manifest["label"].astype(np.uint8)
    inconsistent_people = manifest.groupby("person_id")["label"].nunique()
    if (inconsistent_people > 1).any():
        people = inconsistent_people[inconsistent_people > 1].index.tolist()
        raise DatasetValidationError(f"people have conflicting labels: {people}")
    manifest["eye"] = manifest["eye"].str.upper()
    if not manifest["eye"].isin(["L", "R"]).all():
        raise DatasetValidationError("raw manifest eye values must be L or R")
    verification_values = manifest["diagnosis_verified"].astype(str).str.strip().str.lower()
    if not verification_values.isin(["true", "false"]).all():
        raise DatasetValidationError(
            "raw manifest diagnosis_verified values must be true or false"
        )
    manifest["diagnosis_verified"] = verification_values == "true"
    if "sample_id" in manifest:
        if manifest["sample_id"].isna().any():
            raise DatasetValidationError("raw manifest sample_id cannot be null")
        manifest["sample_id"] = manifest["sample_id"].astype(str).str.strip()
        if not manifest["sample_id"].is_unique:
            raise DatasetValidationError("raw manifest sample_id values must be unique")

    images: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    person_ids: list[str] = []
    eyes: list[str] = []
    geometry: list[np.ndarray] = []
    diagnosis_sources: list[str] = []
    diagnosis_verified: list[bool] = []
    source_image_paths: list[str] = []
    source_image_hashes: list[str] = []
    exclusions: list[dict[str, str]] = []
    base_dir = manifest_path.parent

    for row_index, row in manifest.iterrows():
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        sample_id = str(row.get("sample_id", f"sample-{row_index:04d}"))
        try:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise DatasetValidationError("image could not be read")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            segmentation = segment_iris(image_rgb)
            images.append(rubber_sheet_normalize(image_rgb, segmentation))
            geometry.append(geometric_features(segmentation))
            labels.append(int(row["label"]))
            sample_ids.append(sample_id)
            person_ids.append(str(row["person_id"]))
            eyes.append(str(row["eye"]))
            diagnosis_sources.append(str(row["diagnosis_source"]))
            diagnosis_verified.append(bool(row["diagnosis_verified"]))
            source_image_paths.append(str(image_path.resolve()))
            source_image_hashes.append(sha256_file(image_path))
        except (DatasetValidationError, OSError, ValueError) as error:
            exclusions.append(
                {"sample_id": sample_id, "image_path": str(image_path), "reason": str(error)}
            )

    if not images:
        raise DatasetValidationError("no raw image passed preprocessing")
    dataset = IrisDataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.uint8),
        sample_ids=np.asarray(sample_ids),
        person_ids=np.asarray(person_ids),
        eyes=np.asarray(eyes),
        diagnosis_sources=np.asarray(diagnosis_sources),
        diagnosis_verified=np.asarray(diagnosis_verified, dtype=np.bool_),
        source_image_paths=np.asarray(source_image_paths),
        source_image_hashes=np.asarray(source_image_hashes),
        geometry=np.stack(geometry),
        geometry_names=GEOMETRY_FEATURE_NAMES,
        metadata={
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": sha256_file(manifest_path),
            "sample_count": len(images),
            "excluded_count": len(exclusions),
            "geometry_available": True,
            "person_identity_status": PROVIDED_IDENTITY_STATUS,
            "clinical_label_status": (
                VERIFIED_CLINICAL_LABEL_STATUS
                if manifest["diagnosis_verified"].all()
                else UNVERIFIED_CLINICAL_LABEL_STATUS
            ),
            "positive_label_name": "dm2",
            "negative_label_name": "control",
            "diagnosis_sources": sorted(manifest["diagnosis_source"].unique().tolist()),
            "segmentation_method": "dark-region pupil plus radial-gradient iris boundary",
            "normalization": "Daugman-style rubber sheet, 201x720 RGB",
        },
    )
    dataset.save(output_dir)
    (output_dir / "exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    return dataset
