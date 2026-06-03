from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
NEGATIVE_LABELS = {"not_drone", "no_drone", "non_drone", "background", "noise", "ambient", "negative"}
POSITIVE_LABELS = {"drone", "uav", "quadcopter", "multicopter", "positive"}
SPECIAL_LABELS = {"helicopter", "airplane", "bird", "vehicle", "engine", "fan", "wind"}
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
    "original_dataset_path",
    "split_group",
    "label",
    "label_confidence",
    "label_source",
    "distance_m",
    "distance_category",
    "drone_id",
    "environment",
    "license_notes",
]


def path_tokens(path: Path) -> list[str]:
    tokens: list[str] = []
    for part in path.parts:
        stem = Path(part).stem.lower()
        tokens.extend(token for token in re.split(r"[^a-z0-9]+", stem) if token)
        compact = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
        if compact:
            tokens.append(compact)
    return tokens


def _has_phrase(tokens: list[str], phrase: str) -> bool:
    phrase_tokens = [token for token in re.split(r"[^a-z0-9]+", phrase.lower()) if token]
    if not phrase_tokens:
        return False
    if "_".join(phrase_tokens) in tokens:
        return True
    if len(phrase_tokens) == 1:
        return phrase_tokens[0] in tokens
    for idx in range(0, len(tokens) - len(phrase_tokens) + 1):
        if tokens[idx : idx + len(phrase_tokens)] == phrase_tokens:
            return True
    return False


def infer_label_details(path: Path, source_dataset: str | None = None) -> dict[str, Any]:
    tokens = path_tokens(path)
    name = Path(path).name.upper()
    source = source_dataset or infer_source_dataset(path)

    if source.lower() == "svanstrom":
        if name.startswith("DRONE_"):
            return {"label": "drone", "label_confidence": 1.0, "label_source": "svanstrom_filename"}
        if name.startswith("BACKGROUND_"):
            return {"label": "background", "label_confidence": 1.0, "label_source": "svanstrom_filename"}
        if name.startswith("HELICOPTER_"):
            return {"label": "helicopter", "label_confidence": 1.0, "label_source": "svanstrom_filename"}

    for label in NEGATIVE_LABELS:
        if _has_phrase(tokens, label):
            normalized = "background" if label in {"not_drone", "no_drone", "non_drone", "negative", "ambient"} else label
            return {"label": normalized, "label_confidence": 0.95, "label_source": "negative_path_token"}

    for label in SPECIAL_LABELS:
        if _has_phrase(tokens, label):
            return {"label": label, "label_confidence": 0.90, "label_source": "special_path_token"}

    for label in POSITIVE_LABELS:
        if _has_phrase(tokens, label):
            normalized = "drone" if label in {"uav", "quadcopter", "multicopter", "positive"} else label
            confidence = 0.90 if source in {"DroneAudioSet", "DADS", "Sara", "Acoustic-UAV", "BowonY"} else 0.85
            return {"label": normalized, "label_confidence": confidence, "label_source": "positive_path_token"}

    return {"label": "unknown", "label_confidence": 0.0, "label_source": "unknown"}


def infer_label(path: Path) -> str:
    return str(infer_label_details(path)["label"])


def infer_distance_category(path: Path) -> str | None:
    for part in (component.lower() for component in path.parts):
        if "close" in part:
            return "close"
        if "medium" in part:
            return "medium"
        if "distant" in part or "far" in part:
            return "distant"
    return None


def infer_distance_m(path: Path) -> float | None:
    joined = "/".join(part.lower() for part in path.parts)
    match = re.search(r"(?:mic[-_ ]?dist|distance|dist)[-_ ]?(\d+(?:\.\d+)?)\s*(cm|m)\b", joined)
    if not match:
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(cm|m)\b", joined)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    return value / 100.0 if unit == "cm" else value


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


def infer_drone_id(path: Path) -> str | None:
    joined = "/".join(part.lower() for part in path.parts)
    match = re.search(r"\b(drone\d+|uav\d+|quadcopter\d+|multicopter\d+)\b", joined)
    return match.group(1) if match else None


def infer_environment(path: Path) -> str | None:
    tokens = set(path_tokens(path))
    for environment in ("indoor", "outdoor", "urban", "rural", "field", "lab", "anechoic"):
        if environment in tokens:
            return environment
    return None


def iter_manifest_rows(roots: list[Path]) -> Iterator[dict[str, Any]]:
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            source_dataset = infer_source_dataset(path)
            label_details = infer_label_details(path, source_dataset)
            try:
                original_dataset_path = str(path.relative_to(root))
            except ValueError:
                original_dataset_path = str(path)
            yield {
                "file_path": str(path),
                "file_name": path.name,
                "extension": path.suffix.lower().lstrip("."),
                "source_dataset": source_dataset,
                "original_dataset_path": original_dataset_path,
                "split_group": infer_split_group(path, source_dataset),
                **label_details,
                "distance_m": infer_distance_m(path),
                "distance_category": infer_distance_category(path),
                "drone_id": infer_drone_id(path),
                "environment": infer_environment(path),
                "license_notes": None,
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
