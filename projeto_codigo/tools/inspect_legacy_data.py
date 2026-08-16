from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

from iris_dm2.data import (
    inspect_pickle_globals,
    read_legacy_member,
    restricted_numpy_load,
)


def inspect_archive(archive_path: Path, load_metadata: bool = False) -> dict[str, object]:
    datasets: list[dict[str, object]] = []
    global_references: Counter[str] = Counter()

    with zipfile.ZipFile(archive_path) as archive:
        pickle_names = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".p") and not name.startswith("__MACOSX/")
        )

        for name in pickle_names:
            payload = read_legacy_member(archive, name)
            references, parser_error = inspect_pickle_globals(payload)
            global_references.update(references)
            datasets.append(
                {
                    "path": name,
                    "compressed_bytes": archive.getinfo(name).compress_size,
                    "uncompressed_bytes": len(payload),
                    "global_references": references,
                    "parser_error": parser_error,
                }
            )
            if load_metadata:
                value = restricted_numpy_load(payload)
                datasets[-1]["python_type"] = type(value).__name__
                datasets[-1]["shape"] = list(value.shape)
                datasets[-1]["dtype"] = str(value.dtype)
                datasets[-1]["array_bytes"] = int(value.nbytes)

    return {
        "archive": str(archive_path.resolve()),
        "archive_bytes": archive_path.stat().st_size,
        "pickle_count": len(datasets),
        "global_references": dict(sorted(global_references.items())),
        "pickles": datasets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect legacy pickle opcodes without deserializing them."
    )
    parser.add_argument("archive", type=Path, help="Path to the legacy Data.zip archive")
    parser.add_argument(
        "--load-metadata",
        action="store_true",
        help="Load arrays with a strict NumPy-only global allowlist and report shape/dtype",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(inspect_archive(args.archive, load_metadata=args.load_metadata), indent=2))


if __name__ == "__main__":
    main()
