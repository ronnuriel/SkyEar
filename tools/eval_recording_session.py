from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a local SkyEar recording session for evaluation.")
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    session = Path(args.session)
    metadata_path = session / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"metadata.json not found under {session}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    markers = session / "markers.csv"
    marker_count = max(0, len(markers.read_text(encoding="utf-8").splitlines()) - 1) if markers.exists() else 0
    wav_files = metadata.get("wav_files") or []
    duration = sum(float(item.get("duration_sec") or 0.0) for item in wav_files)
    print(
        json.dumps(
            {
                "session_id": metadata.get("session_id"),
                "station_id": metadata.get("station_id"),
                "wav_count": len(wav_files),
                "marker_count": marker_count,
                "duration_sec": duration,
                "sample_rate": metadata.get("sample_rate"),
                "channels": metadata.get("channels"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
