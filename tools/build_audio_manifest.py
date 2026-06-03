from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tools.dataset_registry import DEFAULT_REGISTRY, dataset_by_id, list_datasets, resolve_local_dir


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
NEGATIVE_LABELS = {"not_drone", "no_drone", "non_drone", "background", "noise", "ambient", "negative"}
POSITIVE_LABELS = {"drone", "uav", "quadcopter", "multicopter", "positive"}
SPECIAL_LABELS = {"helicopter", "airplane", "bird", "vehicle", "engine", "fan", "wind"}
DATASET_MARKERS = {
    "svanstrom": "svanstrom",
    "droneaudioset": "DroneAudioSet",
    "dads": "DADS",
    "sara": "Sara",
    "alemadi": "Sara",
    "acoustic-uav": "Acoustic-UAV",
    "acoustic_uav": "Acoustic-UAV",
    "bowony": "BowonY",
    "field": "field_recordings",
}
FIELDNAMES = [
    "audio_path",
    "file_path",
    "file_name",
    "extension",
    "dataset_id",
    "source_dataset",
    "split",
    "split_group",
    "original_dataset_path",
    "label",
    "original_label",
    "label_confidence",
    "label_source",
    "distance_m",
    "distance_category",
    "drone_id",
    "drone_model",
    "throttle",
    "environment",
    "mic_id",
    "array_info",
    "data_type",
    "sample_rate",
    "duration_sec",
    "channels",
    "license",
    "license_notes",
    "sha256",
    "is_training_candidate",
    "is_benchmark_candidate",
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


def infer_label_details(
    path: Path,
    source_dataset: str | None = None,
    *,
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokens = path_tokens(path)
    name = Path(path).name.upper()
    source = source_dataset or infer_source_dataset(path)
    mapping = {str(k).lower(): v for k, v in (dataset or {}).get("label_mapping", {}).items()}

    if source.lower() == "svanstrom":
        if name.startswith("DRONE_"):
            return {"label": "drone", "original_label": "DRONE", "label_confidence": 1.0, "label_source": "svanstrom_filename"}
        if name.startswith("BACKGROUND_"):
            return {
                "label": "background",
                "original_label": "BACKGROUND",
                "label_confidence": 1.0,
                "label_source": "svanstrom_filename",
            }
        if name.startswith("HELICOPTER_"):
            return {
                "label": "helicopter",
                "original_label": "HELICOPTER",
                "label_confidence": 1.0,
                "label_source": "svanstrom_filename",
            }

    for token in tokens:
        if token in mapping:
            return {
                "label": mapping[token],
                "original_label": token,
                "label_confidence": 0.95,
                "label_source": "registry_label_mapping",
            }

    for label in NEGATIVE_LABELS:
        if _has_phrase(tokens, label):
            normalized = "background" if label in {"not_drone", "no_drone", "non_drone", "negative", "ambient"} else label
            return {
                "label": normalized,
                "original_label": label,
                "label_confidence": 0.95,
                "label_source": "negative_path_token",
            }

    for label in SPECIAL_LABELS:
        if _has_phrase(tokens, label):
            return {"label": label, "original_label": label, "label_confidence": 0.90, "label_source": "special_path_token"}

    for label in POSITIVE_LABELS:
        if _has_phrase(tokens, label):
            normalized = "drone" if label in {"uav", "quadcopter", "multicopter", "positive"} else label
            confidence = 0.90 if source in {"DroneAudioSet", "DADS", "Sara", "Acoustic-UAV", "BowonY"} else 0.85
            return {
                "label": normalized,
                "original_label": label,
                "label_confidence": confidence,
                "label_source": "positive_path_token",
            }

    return {"label": "unknown", "original_label": None, "label_confidence": 0.0, "label_source": "unknown"}


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


def parse_droneaudioset_path(path: Path) -> dict[str, Any]:
    joined = "/".join(part.lower() for part in path.parts)
    metadata: dict[str, Any] = {}
    drone_match = re.search(r"\b(drone\d+)\b", joined)
    if drone_match:
        metadata["drone_id"] = drone_match.group(1)
    distance_match = re.search(r"mic[-_]?dist[-_]?(\d+(?:\.\d+)?)(cm|m)", joined)
    if distance_match:
        value = float(distance_match.group(1))
        unit = distance_match.group(2)
        metadata["distance_m"] = value / 100.0 if unit == "cm" else value
        metadata["distance_category"] = distance_category_from_m(metadata["distance_m"])
    throttle_match = re.search(r"throttle[-_]?([a-z0-9]+)", joined)
    if throttle_match:
        metadata["throttle"] = throttle_match.group(1)
    mic_match = re.search(r"\b(mic\d+)[-_]?([0-9]+array[-_a-z0-9]*)?", joined)
    if mic_match:
        metadata["mic_id"] = mic_match.group(1)
        if mic_match.group(2):
            metadata["array_info"] = re.sub(r"[-_]?file\d+.*$", "", mic_match.group(2))
    if "drone-only" in joined or "drone_only" in joined:
        metadata["data_type"] = "drone-only"
    return metadata


def parse_alemadi_path(path: Path) -> dict[str, Any]:
    joined = "/".join(part.lower() for part in path.parts)
    metadata: dict[str, Any] = {}
    if "binary_drone_audio" in joined or "binary-drone-audio" in joined:
        metadata["data_type"] = "binary_drone_audio"
    if "multiclass_drone_audio" in joined or "multiclass-drone-audio" in joined:
        metadata["data_type"] = "multiclass_drone_audio"
    return metadata


def distance_category_from_m(distance_m: float | None) -> str | None:
    if distance_m is None:
        return None
    if distance_m <= 50:
        return "close"
    if distance_m <= 150:
        return "medium"
    return "distant"


def source_specific_metadata(path: Path, dataset: dict[str, Any] | None, source_dataset: str) -> dict[str, Any]:
    parser = str((dataset or {}).get("metadata_parser") or "").lower()
    if parser == "droneaudioset" or source_dataset == "DroneAudioSet":
        return parse_droneaudioset_path(path)
    if parser == "alemadi" or source_dataset == "Sara":
        return parse_alemadi_path(path)
    return {}


def audio_info(path: Path) -> dict[str, Any]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {"sample_rate": int(info.samplerate), "duration_sec": float(info.duration), "channels": int(info.channels)}
    except Exception:
        if path.suffix.lower() == ".wav":
            try:
                from scipy.io import wavfile

                sample_rate, audio = wavfile.read(str(path), mmap=True)
                channels = 1 if audio.ndim == 1 else int(audio.shape[1])
                return {
                    "sample_rate": int(sample_rate),
                    "duration_sec": float(audio.shape[0] / float(sample_rate)),
                    "channels": channels,
                }
            except Exception:
                return {"sample_rate": None, "duration_sec": None, "channels": None}
        return {"sample_rate": None, "duration_sec": None, "channels": None}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_manifest_rows(
    roots: list[Path],
    *,
    dataset: dict[str, Any] | None = None,
    verify_audio: bool = False,
    skip_audio_hash: bool = True,
    max_files: int | None = None,
) -> Iterator[dict[str, Any]]:
    count = 0
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            if max_files is not None and count >= max_files:
                return
            source_dataset = dataset.get("display_name") if dataset else infer_source_dataset(path)
            source_dataset = source_dataset or infer_source_dataset(path)
            dataset_id = dataset.get("dataset_id") if dataset else None
            label_details = infer_label_details(path, str(source_dataset), dataset=dataset)
            try:
                original_dataset_path = str(path.relative_to(root))
            except ValueError:
                original_dataset_path = str(path)
            split = infer_split_group(path, str(source_dataset))
            metadata = {
                "distance_m": infer_distance_m(path),
                "distance_category": infer_distance_category(path),
                "drone_id": infer_drone_id(path),
                "drone_model": None,
                "throttle": None,
                "environment": infer_environment(path),
                "mic_id": None,
                "array_info": None,
                "data_type": None,
            }
            metadata.update({key: value for key, value in source_specific_metadata(path, dataset, str(source_dataset)).items() if value is not None})
            if metadata["distance_category"] is None:
                metadata["distance_category"] = distance_category_from_m(metadata.get("distance_m"))
            info = audio_info(path) if verify_audio else {"sample_rate": None, "duration_sec": None, "channels": None}
            use_for = set(dataset.get("use_for", []) if dataset else [])
            row = {
                "audio_path": str(path),
                "file_path": str(path),
                "file_name": path.name,
                "extension": path.suffix.lower().lstrip("."),
                "dataset_id": dataset_id or infer_source_dataset(path),
                "source_dataset": source_dataset,
                "split": split,
                "split_group": split,
                "original_dataset_path": original_dataset_path,
                **label_details,
                **metadata,
                **info,
                "license": (dataset or {}).get("license"),
                "license_notes": (dataset or {}).get("license_notes"),
                "sha256": None if skip_audio_hash else sha256_file(path),
                "is_training_candidate": "training_candidate" in use_for,
                "is_benchmark_candidate": "benchmark" in use_for,
            }
            count += 1
            yield {field: row.get(field) for field in FIELDNAMES}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _datasets_from_args(args: argparse.Namespace) -> list[dict[str, Any] | None]:
    if not args.registry:
        return [None]
    if args.dataset:
        return [dataset_by_id(dataset_id, args.registry) for dataset_id in args.dataset]
    return list_datasets(args.registry) if not args.root else [None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SkyEar audio dataset manifest.")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--root", action="append", type=Path, help="Dataset root. Can be passed more than once.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--verify-audio", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--include-manual", action="store_true")
    parser.add_argument("--skip-audio-hash", action="store_true", default=True)
    parser.add_argument("--audio-hash", dest="skip_audio_hash", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.registry is None and not args.root:
        args.registry = DEFAULT_REGISTRY
    rows: list[dict[str, Any]] = []
    for dataset in _datasets_from_args(args):
        if dataset is not None and dataset.get("status") == "manual_download_required" and not args.include_manual:
            continue
        roots = args.root or ([resolve_local_dir(dataset)] if dataset is not None else [])
        rows.extend(
            iter_manifest_rows(
                roots,
                dataset=dataset,
                verify_audio=args.verify_audio,
                skip_audio_hash=args.skip_audio_hash,
                max_files=args.max_files,
            )
        )
    output_csv = args.output or args.output_csv
    if output_csv:
        write_csv(output_csv, rows)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, rows)
    if not output_csv and not args.output_jsonl:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    print(f"manifest_files={len(rows)}")


if __name__ == "__main__":
    main()
