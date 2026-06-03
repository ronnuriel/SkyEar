from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
LABEL_KEYWORDS = ("drone", "helicopter", "airplane", "bird", "background", "noise")
DATASET_MARKERS = {
    "svanstrom": "svanstrom",
    "droneaudioset": "DroneAudioSet",
    "dads": "DADS",
    "sara": "Sara",
    "acoustic-uav": "Acoustic-UAV",
    "acoustic_uav": "Acoustic-UAV",
    "bowony": "BowonY",
    "field": "field_recordings",
}
FIELDNAMES = [
    "file_path",
    "file_name",
    "extension",
    "source_dataset",
    "split_group",
    "label",
    "distance_category",
]


def infer_label(path: Path) -> str:
    joined = "/".join(part.lower() for part in path.parts)
    for label in LABEL_KEYWORDS:
        if label in joined:
            return label
    return "unknown"


def infer_distance_category(path: Path) -> str | None:
    for part in (component.lower() for component in path.parts):
        if "close" in part:
            return "close"
        if "medium" in part:
            return "medium"
        if "distant" in part or "far" in part:
            return "distant"
    return None


def infer_source_dataset(path: Path) -> str:
    joined = "/".join(part.lower() for part in path.parts)
    compact = joined.replace("-", "").replace("_", "")
    if "droneaudioset" in compact:
        return "DroneAudioSet"
    if "acousticuav" in compact:
        return "Acoustic-UAV"
    for marker, dataset in DATASET_MARKERS.items():
        if marker in joined:
            return dataset
    return "unknown"


def infer_split_group(path: Path, source_dataset: str) -> str:
    parts = [part.lower() for part in path.parts]
    for split in ("train", "test", "val", "valid", "validation", "holdout"):
        if split in parts:
            return "validation" if split in {"val", "valid"} else split
    return source_dataset


def iter_manifest_rows(roots: list[Path]) -> Iterator[dict[str, Any]]:
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            source_dataset = infer_source_dataset(path)
            yield {
                "file_path": str(path),
                "file_name": path.name,
                "extension": path.suffix.lower().lstrip("."),
                "source_dataset": source_dataset,
                "split_group": infer_split_group(path, source_dataset),
                "label": infer_label(path),
                "distance_category": infer_distance_category(path),
            }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SkyEar audio dataset manifest.")
    parser.add_argument("--root", action="append", type=Path, required=True, help="Dataset root. Can be passed more than once.")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_manifest_rows(args.root))
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, rows)
    if not args.output_csv and not args.output_jsonl:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    print(f"manifest_files={len(rows)}")


if __name__ == "__main__":
    main()
