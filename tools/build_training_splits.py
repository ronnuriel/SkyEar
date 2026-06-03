from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_key(row: dict[str, Any]) -> str:
    source_path = row.get("original_dataset_path") or row.get("audio_path") or row.get("file_path") or row.get("file_name")
    stem = Path(str(source_path)).stem
    return "|".join(
        [
            str(row.get("dataset_id") or "unknown"),
            stem,
            str(row.get("drone_id") or ""),
            str(row.get("environment") or ""),
        ]
    )


def assign_group(key: str, *, val_pct: float = 0.15, test_pct: float = 0.15) -> str:
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < test_pct:
        return "test"
    if value < test_pct + val_pct:
        return "val"
    return "train"


def build_splits(rows: list[dict[str, Any]], *, val_pct: float = 0.15, test_pct: float = 0.15) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    splits = {"train": [], "val": [], "test": [], "holdout_by_source": []}
    source_seen: set[str] = set()
    for key, items in groups.items():
        split = assign_group(key, val_pct=val_pct, test_pct=test_pct)
        splits[split].extend(items)
        source = str(items[0].get("dataset_id") or "unknown")
        if source not in source_seen:
            splits["holdout_by_source"].extend(items)
            source_seen.add(source)
    return splits


def write_split(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe SkyEar training split manifests.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--val-pct", type=float, default=0.15)
    parser.add_argument("--test-pct", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    fieldnames = list(rows[0].keys()) if rows else []
    splits = build_splits(rows, val_pct=args.val_pct, test_pct=args.test_pct)
    names = {
        "train": "train_manifest.csv",
        "val": "val_manifest.csv",
        "test": "test_manifest.csv",
        "holdout_by_source": "holdout_by_source_manifest.csv",
    }
    for split, filename in names.items():
        write_split(args.output_dir / filename, splits[split], fieldnames)
        print(f"{split}={len(splits[split])}")


if __name__ == "__main__":
    main()
