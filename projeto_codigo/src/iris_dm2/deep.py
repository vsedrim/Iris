from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from iris_dm2.data import (
    IrisDataset,
    require_clinical_label_basis,
    require_identity_basis,
    sha256_file,
)
from iris_dm2.evaluation import (
    METRIC_COLUMNS,
    classification_metrics,
    create_grouped_folds,
)
from iris_dm2.features import PHOTOMETRIC_VARIANTS, apply_photometric_variant

DEEP_MODEL_NAMES = ("resnet18", "efficientnet_b0", "vit_b_16")
DEEP_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAINING_COLOR_JITTER = {
    "brightness": 0.08,
    "contrast": 0.08,
    "saturation": 0.04,
    "hue": 0.01,
}


@dataclass
class DeepEvaluationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    fold_assignments: pd.DataFrame
    summary: pd.DataFrame


class IrisTorchDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        transform,
        variant: str = "original",
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform
        self.variant = variant

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        index = int(self.indices[item])
        image = apply_photometric_variant(self.images[index], self.variant)
        return self.transform(image), int(self.labels[index])


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def derive_run_seed(seed: int, fold: int, model_name: str) -> int:
    payload = f"{seed}:{fold}:{model_name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def set_training_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        ):
            module.eval()


def deep_transform_config() -> dict[str, object]:
    return {
        "training_source_variant": "original",
        "image_size": DEEP_IMAGE_SIZE,
        "color_jitter": dict(TRAINING_COLOR_JITTER),
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "named_photometric_variants_used_for_evaluation_only": True,
    }


def _transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    evaluation = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    training = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ColorJitter(**TRAINING_COLOR_JITTER),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return training, evaluation


def build_deep_model(
    name: str, pretrained: bool = True
) -> tuple[nn.Module, transforms.Compose, transforms.Compose]:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.features[-2:].parameters():
            parameter.requires_grad = True
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    elif name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.encoder.layers[-2:].parameters():
            parameter.requires_grad = True
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    else:
        raise ValueError(f"unknown deep model: {name}")
    training_transform, evaluation_transform = _transforms(image_size=DEEP_IMAGE_SIZE)
    return model, training_transform, evaluation_transform


def _train_model(
    model: nn.Module,
    loader: DataLoader,
    labels: np.ndarray,
    train_indices: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    counts = np.bincount(labels[train_indices], minlength=2).astype(np.float32)
    class_weights = len(train_indices) / (2.0 * np.maximum(counts, 1.0))
    loss_function = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    for _ in range(epochs):
        set_training_mode(model)
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for inputs, _ in loader:
            logits = model(inputs.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def _loader(
    dataset: IrisDataset,
    indices: np.ndarray,
    transform,
    batch_size: int,
    variant: str,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        IrisTorchDataset(dataset.images, dataset.labels, indices, transform, variant),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def evaluate_deep_models(
    dataset: IrisDataset,
    model_names: tuple[str, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pretrained: bool,
    device_name: str,
) -> DeepEvaluationResult:
    unknown_models = set(model_names) - set(DEEP_MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"unknown deep models: {sorted(unknown_models)}")
    unknown_variants = set(variants) - set(PHOTOMETRIC_VARIANTS)
    if unknown_variants:
        raise ValueError(f"unknown photometric variants: {sorted(unknown_variants)}")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else device_name
        if device_name != "auto"
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    assignment_frames: list[pd.DataFrame] = []

    for seed in seeds:
        folds, assignments = create_grouped_folds(
            dataset.labels,
            dataset.person_ids,
            dataset.sample_ids,
            n_splits=n_splits,
            seed=seed,
        )
        assignment_frames.append(assignments)
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            for model_name in model_names:
                run_seed = derive_run_seed(seed, fold_index, model_name)
                set_reproducibility(run_seed)
                model, training_transform, evaluation_transform = build_deep_model(
                    model_name, pretrained=pretrained
                )
                model.to(device)
                training_loader = _loader(
                    dataset,
                    train_indices,
                    training_transform,
                    batch_size,
                    "original",
                    True,
                    run_seed,
                )
                start_time = perf_counter()
                _train_model(
                    model,
                    training_loader,
                    dataset.labels,
                    train_indices,
                    device,
                    epochs,
                    learning_rate,
                )
                fit_seconds = perf_counter() - start_time
                for variant in variants:
                    evaluation_loader = _loader(
                        dataset,
                        test_indices,
                        evaluation_transform,
                        batch_size,
                        variant,
                        False,
                        seed,
                    )
                    predicted = _predict(model, evaluation_loader, device)
                    row: dict[str, object] = {
                        "model": model_name,
                        "variant": variant,
                        "seed": seed,
                        "run_seed": run_seed,
                        "fold": fold_index,
                        "epochs": epochs,
                        "train_samples": len(train_indices),
                        "test_samples": len(test_indices),
                        "fit_seconds": fit_seconds,
                        "device": str(device),
                    }
                    row.update(classification_metrics(dataset.labels[test_indices], predicted))
                    metric_rows.append(row)
                    prediction_rows.extend(
                        {
                            "sample_id": str(dataset.sample_ids[index]),
                            "person_id": str(dataset.person_ids[index]),
                            "label": int(dataset.labels[index]),
                            "prediction": int(prediction),
                            "model": model_name,
                            "variant": variant,
                            "seed": seed,
                            "run_seed": run_seed,
                            "fold": fold_index,
                        }
                        for index, prediction in zip(test_indices, predicted, strict=True)
                    )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    fold_assignments = pd.concat(assignment_frames, ignore_index=True)
    summary = metrics.groupby(["model", "variant"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in METRIC_COLUMNS},
        **{f"{metric}_std": (metric, "std") for metric in METRIC_COLUMNS},
        evaluations=("fold", "count"),
        seeds=("seed", "nunique"),
        folds_per_seed=("fold", "nunique"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )
    return DeepEvaluationResult(metrics, predictions, fold_assignments, summary)


def run_deep_evaluation(
    dataset_path: Path,
    output_dir: Path,
    model_names: tuple[str, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    n_splits: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pretrained: bool,
    device_name: str,
    allow_unverified_identity: bool = False,
    allow_unverified_labels: bool = False,
) -> DeepEvaluationResult:
    dataset = IrisDataset.load(dataset_path)
    require_identity_basis(dataset, allow_unverified=allow_unverified_identity)
    require_clinical_label_basis(dataset, allow_unverified=allow_unverified_labels)
    result = evaluate_deep_models(
        dataset,
        model_names,
        variants,
        seeds,
        n_splits,
        epochs,
        batch_size,
        learning_rate,
        pretrained,
        device_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(output_dir / "deep_metrics_by_fold.csv", index=False)
    result.predictions.to_csv(output_dir / "deep_predictions.csv", index=False)
    result.fold_assignments.to_csv(output_dir / "deep_fold_assignments.csv", index=False)
    result.summary.to_csv(output_dir / "deep_summary.csv", index=False)
    config = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_metadata": dataset.metadata,
        "models": list(model_names),
        "variants": list(variants),
        "seeds": list(seeds),
        "n_splits": n_splits,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pretrained": pretrained,
        "device": device_name,
        "transforms": deep_transform_config(),
        "unverified_identity_explicitly_allowed": allow_unverified_identity,
        "unverified_labels_explicitly_allowed": allow_unverified_labels,
        "metrics": list(METRIC_COLUMNS),
    }
    (output_dir / "deep_run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
