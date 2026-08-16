from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import blob_log, graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import skeletonize

from iris_dm2.data import DatasetValidationError, IrisDataset

FEATURE_GROUPS = ("classic", "color", "vascular", "morphology", "geometry")
PHOTOMETRIC_VARIANTS = (
    "original",
    "brightness_low",
    "brightness_high",
    "contrast_low",
    "contrast_high",
)


@dataclass
class FeatureTable:
    values: np.ndarray
    names: np.ndarray
    groups: np.ndarray
    sample_ids: np.ndarray
    variant: str = "original"

    def validate(self) -> None:
        if self.values.ndim != 2:
            raise DatasetValidationError("feature values must be a two-dimensional matrix")
        if self.values.shape[0] != len(self.sample_ids):
            raise DatasetValidationError("feature rows must match sample IDs")
        if self.values.shape[1] != len(self.names) or len(self.names) != len(self.groups):
            raise DatasetValidationError("feature names and groups must match matrix columns")
        if len(np.unique(self.names)) != len(self.names):
            raise DatasetValidationError("feature names must be unique")
        if not np.all(np.isfinite(self.values)):
            raise DatasetValidationError("feature matrix contains non-finite values")

    def save(self, output_path: Path) -> None:
        self.validate()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            values=self.values,
            names=self.names,
            groups=self.groups,
            sample_ids=self.sample_ids,
            variant=np.asarray(self.variant),
        )

    @classmethod
    def load(cls, path: Path) -> FeatureTable:
        with np.load(path, allow_pickle=False) as archive:
            table = cls(
                values=archive["values"],
                names=archive["names"],
                groups=archive["groups"],
                sample_ids=archive["sample_ids"],
                variant=str(archive["variant"]),
            )
        table.validate()
        return table


def apply_photometric_variant(image_rgb: np.ndarray, variant: str) -> np.ndarray:
    if variant not in PHOTOMETRIC_VARIANTS:
        raise ValueError(f"unknown photometric variant: {variant}")
    if variant == "original":
        return image_rgb
    image = image_rgb.astype(np.float32) / 255.0
    if variant == "brightness_low":
        image *= 0.8
    elif variant == "brightness_high":
        image *= 1.2
    elif variant == "contrast_low":
        image = (image - 0.5) * 0.8 + 0.5
    elif variant == "contrast_high":
        image = (image - 0.5) * 1.2 + 0.5
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def _analysis_image(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(image_rgb, (360, 96), interpolation=cv2.INTER_AREA)
    gray_float = rgb2gray(resized).astype(np.float32)
    gray_uint8 = np.rint(np.clip(gray_float, 0, 1) * 255).astype(np.uint8)
    return resized, gray_uint8


def _classic_features(image_rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    _, gray = _analysis_image(image_rgb)
    pixels = cv2.resize(gray, (64, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    pixel_values = pixels.ravel()
    pixel_names = [
        f"classic_intensity_r{row:02d}_a{column:02d}" for row in range(16) for column in range(64)
    ]

    lbp = local_binary_pattern(gray, P=16, R=2, method="uniform")
    lbp_counts, _ = np.histogram(lbp, bins=18, range=(0, 18))
    lbp_values = lbp_counts.astype(np.float32) / max(lbp_counts.sum(), 1)
    lbp_names = [f"classic_lbp_bin_{index:02d}" for index in range(18)]

    quantized = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    haralick_values: list[float] = []
    haralick_names: list[str] = []
    for property_name in (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ):
        values = graycoprops(glcm, property_name)
        haralick_values.extend([float(values.mean()), float(values.std())])
        normalized_name = property_name.lower()
        haralick_names.extend(
            [f"classic_haralick_{normalized_name}_mean", f"classic_haralick_{normalized_name}_std"]
        )

    values = np.concatenate(
        [pixel_values, lbp_values, np.asarray(haralick_values, dtype=np.float32)]
    )
    return values, pixel_names + lbp_names + haralick_names


def _safe_skew(values: np.ndarray) -> float:
    standard_deviation = float(values.std())
    if standard_deviation < 1e-8:
        return 0.0
    centered = (values - values.mean()) / standard_deviation
    return float(np.mean(centered**3))


def _color_features(image_rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    resized, _ = _analysis_image(image_rgb)
    image_float = resized.astype(np.float32) / 255.0
    spaces = {
        "hsv": (rgb2hsv(image_float), [(0, 1), (0, 1), (0, 1)]),
        "lab": (rgb2lab(image_float), [(0, 100), (-128, 127), (-128, 127)]),
    }
    values: list[float] = []
    names: list[str] = []
    sectors = [("global", slice(None))] + [
        (f"sector_{index}", sector)
        for index, sector in enumerate(np.array_split(np.arange(resized.shape[1]), 4))
    ]
    for space_name, (space, ranges) in spaces.items():
        for channel in range(3):
            flattened = space[:, :, channel].ravel()
            values.extend([float(flattened.mean()), float(flattened.std()), _safe_skew(flattened)])
            names.extend(
                [
                    f"color_{space_name}_c{channel}_mean",
                    f"color_{space_name}_c{channel}_std",
                    f"color_{space_name}_c{channel}_skew",
                ]
            )
        for region_name, columns in sectors:
            region = space[:, columns, :]
            for channel, value_range in enumerate(ranges):
                counts, _ = np.histogram(region[:, :, channel], bins=16, range=value_range)
                normalized = counts.astype(np.float32) / max(counts.sum(), 1)
                values.extend(normalized.tolist())
                names.extend(
                    f"color_{space_name}_{region_name}_c{channel}_hist_{index:02d}"
                    for index in range(16)
                )
    return np.asarray(values, dtype=np.float32), names


def _vascular_features(image_rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    resized, _ = _analysis_image(image_rgb)
    image = resized.astype(np.float32) / 255.0
    green_darkness = 1.0 - image[:, :, 1]
    vesselness = frangi(green_darkness, sigmas=range(1, 4), black_ridges=False)
    if float(vesselness.max()) > 0:
        threshold = float(threshold_otsu(vesselness))
        vessel_mask = vesselness > threshold
    else:
        vessel_mask = np.zeros_like(vesselness, dtype=bool)
    skeleton = skeletonize(vessel_mask)
    neighbors = _angular_periodic_neighbor_count(skeleton)
    branch_points = skeleton & (neighbors >= 4)
    red_green_contrast = (image[:, :, 0] - image[:, :, 1]) / (
        image[:, :, 0] + image[:, :, 1] + 1e-6
    )
    values = np.asarray(
        [
            vesselness.mean(),
            vesselness.std(),
            np.quantile(vesselness, 0.90),
            np.quantile(vesselness, 0.99),
            vessel_mask.mean(),
            skeleton.mean(),
            branch_points.mean(),
            red_green_contrast.mean(),
            red_green_contrast.std(),
            green_darkness.mean(),
            green_darkness.std(),
        ],
        dtype=np.float32,
    )
    names = [
        "vascular_vesselness_mean",
        "vascular_vesselness_std",
        "vascular_vesselness_p90",
        "vascular_vesselness_p99",
        "vascular_mask_density",
        "vascular_skeleton_density",
        "vascular_branch_density",
        "vascular_red_green_contrast_mean",
        "vascular_red_green_contrast_std",
        "vascular_green_darkness_mean",
        "vascular_green_darkness_std",
    ]
    return values, names


def _angular_periodic_neighbor_count(mask: np.ndarray) -> np.ndarray:
    radial_padding = np.pad(mask.astype(np.uint8), ((1, 1), (0, 0)), mode="constant")
    padded = np.pad(radial_padding, ((0, 0), (1, 1)), mode="wrap")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return windows.sum(axis=(-2, -1))


def _morphology_features(image_rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    _, gray = _analysis_image(image_rgb)
    image = gray.astype(np.float32) / 255.0
    radial_gradient = np.gradient(image, axis=0)
    angular_gradient = np.gradient(image, axis=1)
    radial_energy = float(np.mean(radial_gradient**2))
    angular_energy = float(np.mean(angular_gradient**2))
    radial_profile = np.mean(np.abs(radial_gradient), axis=1)
    start = int(len(radial_profile) * 0.20)
    stop = int(len(radial_profile) * 0.65)
    collarette_profile = radial_profile[start:stop]
    collarette_index = start + int(np.argmax(collarette_profile))
    collarette_strength = float(
        (radial_profile[collarette_index] - collarette_profile.mean())
        / (collarette_profile.std() + 1e-6)
    )
    blobs = blob_log(1.0 - image, min_sigma=1, max_sigma=4, num_sigma=4, threshold=0.08)
    blob_density = len(blobs) / image.size
    mean_blob_radius = float(np.mean(blobs[:, 2] * np.sqrt(2))) if len(blobs) else 0.0
    values = np.asarray(
        [
            radial_energy,
            angular_energy,
            angular_energy / (radial_energy + 1e-8),
            collarette_index / len(radial_profile),
            collarette_strength,
            blob_density,
            mean_blob_radius,
            np.mean(np.abs(radial_gradient)),
            np.mean(np.abs(angular_gradient)),
        ],
        dtype=np.float32,
    )
    names = [
        "morphology_radial_gradient_energy",
        "morphology_angular_gradient_energy",
        "morphology_orientation_energy_ratio",
        "morphology_collarette_radius_proxy",
        "morphology_collarette_strength_proxy",
        "morphology_crypt_blob_density_proxy",
        "morphology_crypt_blob_radius_proxy",
        "morphology_radial_edge_mean",
        "morphology_angular_edge_mean",
    ]
    return values, names


EXTRACTORS: dict[str, Callable[[np.ndarray], tuple[np.ndarray, list[str]]]] = {
    "classic": _classic_features,
    "color": _color_features,
    "vascular": _vascular_features,
    "morphology": _morphology_features,
}


def extract_feature_table(
    dataset: IrisDataset,
    feature_groups: tuple[str, ...],
    variant: str = "original",
) -> FeatureTable:
    unknown_groups = set(feature_groups) - set(FEATURE_GROUPS)
    if unknown_groups:
        raise ValueError(f"unknown feature groups: {sorted(unknown_groups)}")
    if "geometry" in feature_groups and not dataset.geometry.size:
        raise DatasetValidationError(
            "geometry was requested but the dataset has no raw-image segmentation metadata"
        )

    rows: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []
    for image_index, image in enumerate(dataset.images):
        transformed = apply_photometric_variant(image, variant)
        row_parts: list[np.ndarray] = []
        row_names: list[str] = []
        row_groups: list[str] = []
        for group in feature_groups:
            if group == "geometry":
                values = dataset.geometry[image_index]
                current_names = [f"geometry_{name}" for name in dataset.geometry_names]
            else:
                values, current_names = EXTRACTORS[group](transformed)
            row_parts.append(np.asarray(values, dtype=np.float32))
            row_names.extend(current_names)
            row_groups.extend([group] * len(current_names))
        if image_index == 0:
            names = row_names
            groups = row_groups
        elif row_names != names:
            raise DatasetValidationError("feature schema changed between samples")
        rows.append(np.concatenate(row_parts))

    table = FeatureTable(
        values=np.stack(rows).astype(np.float32),
        names=np.asarray(names),
        groups=np.asarray(groups),
        sample_ids=dataset.sample_ids.copy(),
        variant=variant,
    )
    table.validate()
    return table


def extract_and_save_variants(
    dataset_path: Path,
    output_dir: Path,
    feature_groups: tuple[str, ...],
    variants: tuple[str, ...],
) -> list[Path]:
    dataset = IrisDataset.load(dataset_path)
    output_paths: list[Path] = []
    for variant in variants:
        table = extract_feature_table(dataset, feature_groups, variant=variant)
        output_path = output_dir / f"features_{variant}.npz"
        table.save(output_path)
        output_paths.append(output_path)
    return output_paths
