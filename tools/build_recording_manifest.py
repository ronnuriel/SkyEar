from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "wav_path",
    "station_id",
    "session_id",
    "start_time",
    "end_time",
    "channels",
    "sample_rate",
    "label_from_markers",
    "distance_m",
    "drone_model",
    "note",
]


def build_manifest(root: str | Path, output: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        session_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        markers = _read_markers(session_dir / "markers.csv")
        for wav in metadata.get("wav_files") or []:
            marker = _nearest_marker(markers, float(wav.get("start_time") or 0.0), float(wav.get("end_time") or 0.0))
            rows.append(
                {
                    "wav_path": wav.get("wav_path"),
                    "station_id": metadata.get("station_id"),
                    "session_id": metadata.get("session_id"),
                    "start_time": wav.get("start_time"),
                    "end_time": wav.get("end_time"),
                    "channels": wav.get("channels") or metadata.get("channels"),
                    "sample_rate": wav.get("sample_rate") or metadata.get("sample_rate"),
                    "label_from_markers": marker.get("label", "") if marker else "",
                    "distance_m": marker.get("distance_m", "") if marker else "",
                    "drone_model": marker.get("drone_model", "") if marker else "",
                    "note": marker.get("note", "") if marker else "",
                }
            )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_markers(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nearest_marker(markers: list[dict[str, str]], start_time: float, end_time: float) -> dict[str, str] | None:
    if not markers:
        return None
    inside = [
        marker
        for marker in markers
        if marker.get("timestamp_unix")
        and start_time <= float(marker["timestamp_unix"]) <= end_time
    ]
    if inside:
        return inside[0]
    return markers[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest from local SkyEar recording sessions.")
    parser.add_argument("--root", default="runtime/recordings")
    parser.add_argument("--output", default="data/manifests/local_recordings_manifest.csv")
    args = parser.parse_args()
    rows = build_manifest(args.root, args.output)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
