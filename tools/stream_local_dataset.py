from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import requests

from shared.event_schema import AcousticEvent, EventStatus, GeoPoint
from station.audio_capture import to_mono
from station.detector_state import StationDetectorState, StationDetectorStateConfig
from station.hf_detector import DEFAULT_MODEL_ID, HFDetector
from station.spectrum import compute_harmonic_lines, compute_spectrogram_summary, compute_spectrum_summary
from tools.stream_hf_dataset import iter_audio_windows, mono_to_simulated_channels, resample_mono


DETECTOR_VERSION = "local-distance-dataset-stream-v1"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
LABEL_KEYWORDS = ("drone", "helicopter", "airplane", "bird", "background", "noise")


def find_audio_files(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def infer_path_metadata(path: Path, root: Path | None = None) -> dict[str, Any]:
    path = Path(path)
    try:
        rel_path = path.relative_to(root) if root is not None else path
    except ValueError:
        rel_path = path

    parts = [part.lower() for part in rel_path.parts]
    metadata: dict[str, Any] = {
        "file_path": str(path),
        "label": "unknown",
        "distance_category": None,
    }

    for part in parts:
        if "close" in part:
            metadata["distance_category"] = "Close"
            break
        if "medium" in part:
            metadata["distance_category"] = "Medium"
            break
        if "distant" in part or "far" in part:
            metadata["distance_category"] = "Distant"
            break

    joined = "/".join(parts)
    for label in LABEL_KEYWORDS:
        if label in joined:
            metadata["label"] = label
            break

    return metadata


def metadata_matches(metadata: dict[str, Any], label_filter: str | None, distance_filter: str | None) -> bool:
    if label_filter and label_filter.lower() not in str(metadata.get("label", "")).lower():
        return False
    if distance_filter and distance_filter.lower() != str(metadata.get("distance_category", "")).lower():
        return False
    return True


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio.reshape(-1)
    if audio.ndim != 2:
        raise ValueError(f"audio must be mono or 2D, got shape {audio.shape}")
    if audio.shape[0] <= 16 and audio.shape[0] < audio.shape[1]:
        return audio.mean(axis=0).astype(np.float32)
    return to_mono(audio)


def read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        return _to_mono_float32(audio), int(sample_rate)
    except Exception as soundfile_error:
        try:
            import librosa

            audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
            return _to_mono_float32(audio), int(sample_rate)
        except Exception as librosa_error:
            try:
                from scipy.io import wavfile

                sample_rate, audio = wavfile.read(str(path))
                audio = np.asarray(audio)
                if np.issubdtype(audio.dtype, np.integer):
                    max_value = float(np.iinfo(audio.dtype).max)
                    audio = audio.astype(np.float32) / max_value
                return _to_mono_float32(audio), int(sample_rate)
            except Exception as scipy_error:
                raise RuntimeError(
                    f"Could not read audio file {path}: {soundfile_error}; {librosa_error}; {scipy_error}"
                ) from scipy_error


def detector_state_from_args(args: argparse.Namespace) -> StationDetectorState:
    state = StationDetectorState(
        StationDetectorStateConfig(
            calibration_seconds=0.0,
            min_suspect_threshold=16.0,
            min_alert_threshold=22.0,
        )
    )
    state.calibrated = True
    return state


def build_event(
    *,
    station_id: str,
    root: Path,
    file_path: Path,
    label: str,
    distance_category: str | None,
    audio: np.ndarray,
    sample_rate: int,
    timestamp: float,
    detector_state: StationDetectorState,
    hf_result: Any = None,
    hf_p_drone: float | None = None,
) -> AcousticEvent:
    frame = detector_state.update(audio, sample_rate, timestamp, hf_p_drone=hf_p_drone)
    mono = to_mono(audio)
    max_freq = detector_state.config.max_freq
    spectrum = compute_spectrum_summary(mono, sample_rate, max_freq=max_freq, n_points=300)
    spectrogram = compute_spectrogram_summary(
        mono,
        sample_rate,
        max_freq=min(5000, max_freq),
        n_freq_bins=96,
        n_time_bins=64,
    )
    harmonic_lines = compute_harmonic_lines(frame.best_f0_hz, max_freq)

    return AcousticEvent(
        station_id=station_id,
        station_name=f"Local Dataset {Path(root).name}",
        timestamp_unix=timestamp,
        station_location=GeoPoint(latitude=32.0853, longitude=34.7818, altitude_m=0.0),
        status=EventStatus(frame.status),
        confidence=frame.confidence,
        harmonic_score=frame.harmonic_score,
        best_f0_hz=frame.best_f0_hz,
        hf_p_drone=hf_p_drone,
        rms=frame.rms,
        peak=frame.peak,
        duration_sec=frame.duration_sec,
        calibrated=frame.calibrated,
        strongest_channel=frame.strongest_channel,
        channel_agreement_count=frame.agreement_count,
        channel_count=frame.channel_count,
        channel_evidence=[item.__dict__ for item in frame.per_channel],
        detector_version=DETECTOR_VERSION,
        station_mode="unsynchronized_multimic_voting" if frame.channel_count > 1 else "mono",
        metadata={
            "source": "local_distance_dataset",
            "dataset_root": str(root),
            "file_path": str(file_path),
            "label": label,
            "distance_category": distance_category,
            "sample_rate": sample_rate,
            "channels": frame.channel_count,
            "mic_profile": "simulated_multichannel_from_dataset" if frame.channel_count > 1 else "mono_dataset_audio",
            "mic_sync_mode": "unsynchronized" if frame.channel_count > 1 else "mono",
            "suspect_threshold": frame.suspect_threshold,
            "alert_threshold": frame.alert_threshold,
            "f0_stable": frame.f0_stable,
            "hf_label": getattr(hf_result, "label", None),
            "hf_class_probs": getattr(hf_result, "class_probs", {}) if hf_result is not None else {},
            "hf_error": getattr(hf_result, "error", None),
            **spectrum,
            **spectrogram,
            "harmonic_lines": harmonic_lines,
        },
    )


def _report_rows(events: list[AcousticEvent]) -> Iterator[dict[str, Any]]:
    for event in events:
        metadata = event.metadata
        yield {
            "file_path": metadata.get("file_path"),
            "label": metadata.get("label"),
            "distance_category": metadata.get("distance_category"),
            "status": event.status.value if hasattr(event.status, "value") else event.status,
            "confidence": event.confidence,
            "hf_p_drone": event.hf_p_drone,
            "hf_label": metadata.get("hf_label"),
            "harmonic_score": event.harmonic_score,
            "best_f0_hz": event.best_f0_hz,
            "f0_stable": metadata.get("f0_stable"),
        }


def write_report(path: Path, events: list[AcousticEvent]) -> None:
    fieldnames = [
        "file_path",
        "label",
        "distance_category",
        "status",
        "confidence",
        "hf_p_drone",
        "hf_label",
        "harmonic_score",
        "best_f0_hz",
        "f0_stable",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_report_rows(events))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream local distance-labeled audio datasets into SkyEar as station events."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--station-id", default="local_dataset_001")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--label-filter")
    parser.add_argument("--distance-filter")
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--save-report", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    detector_state = detector_state_from_args(args)
    hf_detector = HFDetector(model_id=args.model_id) if args.hf else None
    last_hf_result = None
    files_seen = 0
    windows_sent = 0
    events: list[AcousticEvent] = []

    for path in find_audio_files(root):
        metadata = infer_path_metadata(path, root=root)
        if not metadata_matches(metadata, args.label_filter, args.distance_filter):
            continue
        if files_seen >= int(args.max_files):
            break

        mono, source_sr = read_audio_mono(path)
        mono = resample_mono(mono, source_sr, int(args.sample_rate))
        files_seen += 1

        for window in iter_audio_windows(mono, int(args.sample_rate), float(args.window_sec)):
            if args.max_windows is not None and windows_sent >= int(args.max_windows):
                if args.save_report:
                    write_report(args.save_report, events)
                return

            loop_start = time.time()
            audio = mono_to_simulated_channels(window, int(args.channels))
            if hf_detector is not None:
                last_hf_result = hf_detector.predict(window, int(args.sample_rate))
            hf_p_drone = last_hf_result.p_drone if last_hf_result is not None else None
            event = build_event(
                station_id=args.station_id,
                root=root,
                file_path=path,
                label=str(metadata.get("label")),
                distance_category=metadata.get("distance_category"),
                audio=audio,
                sample_rate=int(args.sample_rate),
                timestamp=loop_start,
                detector_state=detector_state,
                hf_result=last_hf_result,
                hf_p_drone=hf_p_drone,
            )
            requests.post(args.server, json=event.model_dump(mode="json"), timeout=2.0)
            events.append(event)
            print(
                f"{path.name} {metadata.get('label')} {metadata.get('distance_category')} {event.status:11s} "
                f"hf_p={event.hf_p_drone} harm={event.harmonic_score:.1f} "
                f"f0={event.best_f0_hz} stable={event.metadata.get('f0_stable')}"
            )
            windows_sent += 1
            if args.realtime:
                time.sleep(max(0.0, float(args.window_sec) - (time.time() - loop_start)))

    if args.save_report:
        write_report(args.save_report, events)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
