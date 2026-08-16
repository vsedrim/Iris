from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
deep_module = pytest.importorskip("iris_dm2.deep")
build_deep_model = deep_module.build_deep_model
deep_transform_config = deep_module.deep_transform_config
derive_run_seed = deep_module.derive_run_seed
set_training_mode = deep_module.set_training_mode


@pytest.mark.parametrize("model_name", ["resnet18", "efficientnet_b0", "vit_b_16"])
def test_deep_model_has_binary_head_and_trainable_upper_layers(model_name: str) -> None:
    model, _, evaluation_transform = build_deep_model(model_name, pretrained=False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    assert trainable
    assert frozen
    if model_name == "resnet18":
        assert model.fc.out_features == 2
    elif model_name == "efficientnet_b0":
        assert model.classifier[-1].out_features == 2
    else:
        assert model.heads.head.out_features == 2
    assert evaluation_transform is not None


def test_deep_run_seed_is_stable_and_independent_per_model() -> None:
    assert derive_run_seed(42, 0, "resnet18") == derive_run_seed(42, 0, "resnet18")
    assert derive_run_seed(42, 0, "resnet18") != derive_run_seed(
        42, 0, "efficientnet_b0"
    )
    assert derive_run_seed(42, 0, "resnet18") != derive_run_seed(42, 1, "resnet18")


def test_deep_transform_config_records_training_augmentation() -> None:
    config = deep_transform_config()

    assert config["training_source_variant"] == "original"
    assert config["image_size"] == 224
    assert config["color_jitter"] == {
        "brightness": 0.08,
        "contrast": 0.08,
        "saturation": 0.04,
        "hue": 0.01,
    }
    assert config["named_photometric_variants_used_for_evaluation_only"] is True


def test_color_jitter_is_applied_only_to_training_transform() -> None:
    _, training_transform, evaluation_transform = build_deep_model(
        "resnet18", pretrained=False
    )

    color_jitter_type = deep_module.transforms.ColorJitter
    assert any(
        isinstance(transform, color_jitter_type)
        for transform in training_transform.transforms
    )
    assert not any(
        isinstance(transform, color_jitter_type)
        for transform in evaluation_transform.transforms
    )


def test_frozen_batch_norm_stays_in_evaluation_mode() -> None:
    model, _, _ = build_deep_model("resnet18", pretrained=False)

    set_training_mode(model)

    assert model.bn1.training is False
    assert model.layer4[0].bn1.training is True
