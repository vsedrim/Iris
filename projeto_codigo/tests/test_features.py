from __future__ import annotations

import numpy as np
import pytest

from iris_dm2.data import DatasetValidationError, IrisDataset
from iris_dm2.features import _angular_periodic_neighbor_count, extract_feature_table


def textured_dataset() -> IrisDataset:
    random = np.random.default_rng(42)
    images = random.integers(0, 256, size=(4, 64, 128, 3), dtype=np.uint8)
    return IrisDataset(
        images=images,
        labels=np.asarray([0, 0, 1, 1], dtype=np.uint8),
        sample_ids=np.asarray(["c0", "c1", "d0", "d1"]),
        person_ids=np.asarray(["pc0", "pc1", "pd0", "pd1"]),
        eyes=np.asarray(["L", "R", "L", "R"]),
    )


def test_all_image_feature_families_are_finite_and_named() -> None:
    table = extract_feature_table(
        textured_dataset(), ("classic", "color", "vascular", "morphology")
    )

    assert table.values.shape[0] == 4
    assert table.values.shape[1] == len(table.names)
    assert np.all(np.isfinite(table.values))
    assert set(table.groups) == {"classic", "color", "vascular", "morphology"}
    assert any("haralick" in name for name in table.names)
    assert any("collarette" in name for name in table.names)
    assert any("vesselness" in name for name in table.names)


def test_photometric_variant_preserves_schema_and_changes_values() -> None:
    dataset = textured_dataset()
    original = extract_feature_table(dataset, ("color",), variant="original")
    darker = extract_feature_table(dataset, ("color",), variant="brightness_low")

    np.testing.assert_array_equal(original.names, darker.names)
    assert not np.allclose(original.values, darker.values)


def test_geometry_cannot_be_fabricated_for_legacy_dataset() -> None:
    with pytest.raises(DatasetValidationError, match="no raw-image segmentation metadata"):
        extract_feature_table(textured_dataset(), ("geometry",))


def test_vascular_neighbors_wrap_angularly_but_not_radially() -> None:
    mask = np.zeros((3, 5), dtype=bool)
    mask[1, 0] = True
    mask[1, -1] = True
    mask[0, 2] = True
    mask[-1, 2] = True

    neighbors = _angular_periodic_neighbor_count(mask)

    assert neighbors[1, 0] == 2
    assert neighbors[0, 2] == 1
