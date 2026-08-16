from __future__ import annotations

import argparse
from pathlib import Path

from iris_dm2 import __version__
from iris_dm2.data import prepare_legacy_archive
from iris_dm2.evaluation import MODEL_NAMES, run_classical_evaluation
from iris_dm2.features import FEATURE_GROUPS, PHOTOMETRIC_VARIANTS, extract_and_save_variants
from iris_dm2.preprocessing import prepare_raw_manifest


def prepare_command(args: argparse.Namespace) -> None:
    dataset = prepare_legacy_archive(
        args.archive,
        args.output,
        args.layout,
        acknowledge_unverified_metadata=args.acknowledge_unverified_legacy_metadata,
    )
    positive_name = dataset.metadata["positive_label_name"]
    negative_name = dataset.metadata["negative_label_name"]
    print(
        f"Prepared {len(dataset.images)} samples "
        f"({int(dataset.labels.sum())} {positive_name}, "
        f"{int((dataset.labels == 0).sum())} {negative_name}) "
        f"in {args.output}"
    )


def preprocess_command(args: argparse.Namespace) -> None:
    dataset = prepare_raw_manifest(args.manifest, args.output)
    print(f"Preprocessed {len(dataset.images)} raw images in {args.output}")


def extract_command(args: argparse.Namespace) -> None:
    paths = extract_and_save_variants(
        args.dataset,
        args.output,
        tuple(args.feature_groups),
        tuple(args.variants),
    )
    print(f"Wrote {len(paths)} feature matrices to {args.output}")


def evaluate_command(args: argparse.Namespace) -> None:
    result = run_classical_evaluation(
        args.dataset,
        args.features,
        args.output,
        tuple(args.models),
        tuple(args.feature_sets),
        tuple(args.variants),
        tuple(args.seeds),
        args.folds,
        args.allow_unverified_identity,
        args.allow_unverified_labels,
    )
    print(f"Wrote {len(result.metrics)} fold-level evaluations to {args.output}")


def deep_command(args: argparse.Namespace) -> None:
    try:
        from iris_dm2.deep import run_deep_evaluation
    except ImportError as error:
        message = "Install deep-learning dependencies with: pip install -e '.[deep]'"
        raise SystemExit(message) from error
    result = run_deep_evaluation(
        args.dataset,
        args.output,
        tuple(args.models),
        tuple(args.variants),
        tuple(args.seeds),
        args.folds,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        not args.no_pretrained,
        args.device,
        args.allow_unverified_identity,
        args.allow_unverified_labels,
    )
    print(f"Wrote {len(result.metrics)} deep fold-level evaluations to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iris-dm2",
        description="Run iris-image association experiments with recorded provenance.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Convert the reviewed legacy archive to an NPZ dataset without pickle"
    )
    prepare_parser.add_argument("--archive", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument(
        "--layout",
        choices=["personBase", "personBase_invert"],
        default="personBase_invert",
    )
    prepare_parser.add_argument(
        "--acknowledge-unverified-legacy-metadata",
        action="store_true",
        help="Acknowledge that source identity and DM2 subtype cannot be verified",
    )
    prepare_parser.set_defaults(handler=prepare_command)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Segment and normalize raw images from an explicit manifest"
    )
    preprocess_parser.add_argument("--manifest", type=Path, required=True)
    preprocess_parser.add_argument("--output", type=Path, required=True)
    preprocess_parser.set_defaults(handler=preprocess_command)

    extract_parser = subparsers.add_parser(
        "extract", help="Extract selected feature families from a prepared dataset"
    )
    extract_parser.add_argument("--dataset", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=FEATURE_GROUPS,
        default=["classic", "color", "vascular", "morphology"],
    )
    extract_parser.add_argument(
        "--variants",
        nargs="+",
        choices=PHOTOMETRIC_VARIANTS,
        default=["original"],
    )
    extract_parser.set_defaults(handler=extract_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run grouped classical-model evaluation and write fold-level reports"
    )
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    evaluate_parser.add_argument("--features", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    evaluate_parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["all", "classic", "color", "vascular", "morphology"],
        help="Feature groups or '+' combinations, for example classic+color",
    )
    evaluate_parser.add_argument(
        "--variants", nargs="+", choices=PHOTOMETRIC_VARIANTS, default=["original"]
    )
    evaluate_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2025])
    evaluate_parser.add_argument("--folds", type=int, default=5)
    evaluate_parser.add_argument(
        "--allow-unverified-identity",
        action="store_true",
        help="Permit provisional row-based grouping when real subject IDs are unavailable",
    )
    evaluate_parser.add_argument(
        "--allow-unverified-labels",
        action="store_true",
        help="Permit provisional evaluation when DM2/control labels are not verified",
    )
    evaluate_parser.set_defaults(handler=evaluate_command)

    deep_parser = subparsers.add_parser(
        "deep", help="Run grouped transfer-learning evaluation with the same split protocol"
    )
    deep_parser.add_argument("--dataset", type=Path, required=True)
    deep_parser.add_argument("--output", type=Path, required=True)
    deep_parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet18", "efficientnet_b0", "vit_b_16"],
        default=["resnet18", "efficientnet_b0", "vit_b_16"],
    )
    deep_parser.add_argument(
        "--variants", nargs="+", choices=PHOTOMETRIC_VARIANTS, default=["original"]
    )
    deep_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2025])
    deep_parser.add_argument("--folds", type=int, default=5)
    deep_parser.add_argument("--epochs", type=int, default=10)
    deep_parser.add_argument("--batch-size", type=int, default=16)
    deep_parser.add_argument("--learning-rate", type=float, default=1e-4)
    deep_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    deep_parser.add_argument("--no-pretrained", action="store_true")
    deep_parser.add_argument(
        "--allow-unverified-identity",
        action="store_true",
        help="Permit provisional row-based grouping when real subject IDs are unavailable",
    )
    deep_parser.add_argument(
        "--allow-unverified-labels",
        action="store_true",
        help="Permit provisional evaluation when DM2/control labels are not verified",
    )
    deep_parser.set_defaults(handler=deep_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
