from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def summarize_recording_session(session_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir)
    metadata_path = session / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    wav_files = _wav_files(session, metadata)
    wav_duration = sum(_wav_duration_sec(path) for path in wav_files)
    wall_duration = _wall_duration_sec(metadata, wav_duration)
    markers = _marker_count(session / "markers.csv")
    overflow_count = int(metadata.get("overflow_count") or len(metadata.get("overflow_timestamps") or []))
    discontinuities = metadata.get("discontinuities") or []
    continuity_ok = bool(metadata.get("recording_continuity_ok", not discontinuities and overflow_count == 0))
    return {
        "session_dir": str(session),
        "session_id": metadata.get("session_id") or session.name,
        "total_wav_duration_sec": float(wav_duration),
        "wall_duration_sec": float(wall_duration),
        "duration_diff_sec": float(wall_duration - wav_duration),
        "marker_count": int(markers),
        "overflow_count": int(overflow_count),
        "discontinuity_count": len(discontinuities),
        "recording_continuity_ok": continuity_ok,
        "wav_count": len(wav_files),
    }


def latest_recording_dir(root: str | Path) -> Path:
    root_path = Path(root)
    candidates = [path for path in root_path.iterdir() if path.is_dir() and (path / "metadata.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No recording sessions found under {root_path}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a SkyEar recording session.")
    parser.add_argument("--root", default="runtime/recordings")
    parser.add_argument("--session", help="Session directory. Defaults to latest under --root.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = Path(args.session) if args.session else latest_recording_dir(args.root)
    summary = summarize_recording_session(session)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"session={summary['session_id']}")
        print(f"folder={summary['session_dir']}")
        print(f"wav_count={summary['wav_count']}")
        print(f"total_wav_duration_sec={summary['total_wav_duration_sec']:.3f}")
        print(f"wall_duration_sec={summary['wall_duration_sec']:.3f}")
        print(f"duration_diff_sec={summary['duration_diff_sec']:.3f}")
        print(f"marker_count={summary['marker_count']}")
        print(f"overflow_count={summary['overflow_count']}")
        print(f"discontinuity_count={summary['discontinuity_count']}")
        print(f"recording_continuity_ok={summary['recording_continuity_ok']}")
    return 0


def _wav_files(session: Path, metadata: dict[str, Any]) -> list[Path]:
    paths = []
    for item in metadata.get("wav_files") or []:
        path = Path(str(item.get("wav_path") or ""))
        if not path.exists() and not path.is_absolute():
            path = session / path
        if path.exists():
            paths.append(path)
    if paths:
        return paths
    return sorted(session.glob("*.wav"))


def _wav_duration_sec(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        from scipy.io import wavfile

        sample_rate, data = wavfile.read(path)
        return float(data.shape[0]) / float(sample_rate)


def _wall_duration_sec(metadata: dict[str, Any], fallback: float) -> float:
    wavs = metadata.get("wav_files") or []
    starts = [float(item["start_time"]) for item in wavs if item.get("start_time") is not None]
    ends = [float(item["end_time"]) for item in wavs if item.get("end_time") is not None]
    if starts and ends:
        return max(0.0, max(ends) - min(starts))
    start = metadata.get("started_unix")
    stop = metadata.get("stopped_unix")
    if start is not None and stop is not None:
        return max(0.0, float(stop) - float(start))
    return float(fallback)


def _marker_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _row in csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
