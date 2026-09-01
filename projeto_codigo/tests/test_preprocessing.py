from __future__ import annotations

import numpy as np

from iris_dm2.preprocessing import (
    Circle,
    SegmentationResult,
    geometric_features,
    rubber_sheet_normalize,
    segment_iris,
    validate_segmentation,
)


def synthetic_eye(
    pupil_center: tuple[int, int] = (160, 160),
    iris_center: tuple[int, int] = (160, 160),
    iris_axes: tuple[int, int] = (105, 105),
) -> np.ndarray:
    size = 320
    y, x = np.ogrid[:size, :size]
    image = np.full((size, size, 3), 235, dtype=np.uint8)
    iris = (
        ((x - iris_center[0]) / iris_axes[0]) ** 2
        + ((y - iris_center[1]) / iris_axes[1]) ** 2
        <= 1
    )
    pupil = np.sqrt(
        (x - pupil_center[0]) ** 2 + (y - pupil_center[1]) ** 2
    ) <= 42
    image[iris] = [75, 125, 155]
    image[pupil] = [8, 8, 8]
    return image


def test_segment_and_normalize_synthetic_eye() -> None:
    image = synthetic_eye()

    segmentation = segment_iris(image)
    normalized = rubber_sheet_normalize(image, segmentation, output_shape=(64, 256))

    assert abs(segmentation.pupil.radius - 42) < 5
    assert abs(segmentation.iris.radius - 105) < 8
    assert normalized.shape == (64, 256, 3)
    assert np.any(normalized[:, -1] != 0)


def test_geometric_features_are_finite_and_ratio_is_physical() -> None:
    segmentation = segment_iris(synthetic_eye())

    features = geometric_features(segmentation)

    assert features.shape == (19,)
    assert np.all(np.isfinite(features))
    assert 0 < features[2] < 1
    assert np.isclose(features[3], features[2] ** 2, rtol=0.15)
    assert np.isclose(features[15], 63, atol=10)
    assert features[16] < 5
    assert np.isclose(features[17], 0.6, atol=0.1)


def test_geometry_detects_independent_centers_and_outer_border_deviation() -> None:
    image = synthetic_eye(
        pupil_center=(150, 160), iris_center=(162, 158), iris_axes=(110, 96)
    )

    segmentation = segment_iris(image)
    features = geometric_features(segmentation)

    assert np.hypot(segmentation.iris.x - 162, segmentation.iris.y - 158) < 10
    assert features[4] > 0.04
    assert not np.isclose(features[3], features[2] ** 2, rtol=0.01)
    assert features[11] > 0.01
    assert segmentation.quality > 0.30


def test_segmentation_quality_control_rejects_low_quality_result() -> None:
    segmentation = segment_iris(synthetic_eye())
    low_quality = SegmentationResult(
        pupil=segmentation.pupil,
        iris=segmentation.iris,
        pupil_contour=segmentation.pupil_contour,
        iris_contour=segmentation.iris_contour,
        quality=0.01,
    )

    with np.testing.assert_raises_regex(ValueError, "segmentation quality"):
        validate_segmentation(low_quality, (320, 320))


def test_segmentation_quality_control_rejects_crossing_boundaries() -> None:
    segmentation = segment_iris(synthetic_eye())
    crossing = SegmentationResult(
        pupil=Circle(160, 160, 42),
        iris=Circle(160, 160, 45),
        pupil_contour=segmentation.pupil_contour,
        iris_contour=segmentation.pupil_contour.copy(),
        quality=0.9,
    )

    with np.testing.assert_raises_regex(ValueError, "annulus|crosses"):
        validate_segmentation(crossing, (320, 320))
