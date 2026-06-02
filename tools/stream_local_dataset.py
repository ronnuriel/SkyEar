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
from tools.stream_hf_dataset import format_detection_log, iter_audio_windows, mono_to_simulated_channels, resample_mono


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
            metadata["distance_category"] = "close"
            break
        if "medium" in part:
            metadata["distance_category"] = "medium"
            break
        if "distant" in part or "far" in part:
            metadata["distance_category"] = "distant"
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


def iter_audio_windows_with_padding(
    mono: np.ndarray,
    sample_rate: int,
    window_sec: float,
    skip_tail_padding: bool = True,
) -> Iterator[tuple[np.ndarray, float]]:
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    window_samples = max(1, int(round(float(sample_rate) * float(window_sec))))
    for start in range(0, mono.size, window_samples):
        actual = min(window_samples, mono.size - start)
        padding_ratio = 1.0 - (actual / window_samples)
        if skip_tail_padding and padding_ratio > 0.20:
            continue
        window = mono[start : start + window_samples]
        if window.size < window_samples:
            window = np.pad(window, (0, window_samples - window.size))
        yield window.astype(np.float32), float(padding_ratio)


def _window_rms(mono: np.ndarray) -> float:
    if mono.size == 0:
        return 0.0
    audio64 = np.asarray(mono, dtype=np.float64)
    return float(np.sqrt(np.mean(audio64 * audio64)))


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
        harmonic_score_smoothed=frame.harmonic_score_smoothed,
        harmonic_evidence_pct=frame.harmonic_evidence_pct,
        harmonic_evidence_pct_smoothed=frame.harmonic_evidence_pct_smoothed,
        best_f0_hz=frame.best_f0_hz,
        raw_best_f0_hz=frame.raw_best_f0_hz,
        canonical_best_f0_hz=frame.canonical_best_f0_hz,
        f0_family_stable=frame.f0_family_stable,
        ml_drone_pct=frame.ml_drone_pct,
        ml_drone_pct_smoothed=frame.ml_drone_pct_smoothed,
        combined_drone_evidence_pct=frame.combined_drone_evidence_pct,
        hf_p_drone=hf_p_drone,
        hf_negative=frame.hf_negative,
        hf_positive=frame.hf_positive,
        decision_reason=frame.decision_reason,
        operator_label=frame.operator_label,
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
            "f0_family_stable": frame.f0_family_stable,
            "raw_best_f0_hz": frame.raw_best_f0_hz,
            "canonical_best_f0_hz": frame.canonical_best_f0_hz,
            "harmonic_score_smoothed": frame.harmonic_score_smoothed,
            "harmonic_evidence_pct": frame.harmonic_evidence_pct,
            "harmonic_evidence_pct_raw": frame.harmonic_evidence_pct_raw,
            "harmonic_evidence_pct_smoothed": frame.harmonic_evidence_pct_smoothed,
            "ml_drone_pct": frame.ml_drone_pct,
            "ml_drone_pct_smoothed": frame.ml_drone_pct_smoothed,
            "combined_drone_evidence_pct": frame.combined_drone_evidence_pct,
            "hf_negative": frame.hf_negative,
            "hf_positive": frame.hf_positive,
            "decision_reason": frame.decision_reason,
            "operator_label": frame.operator_label,
            "hf_label": getattr(hf_result, "label", None),
            "hf_class_probs": getattr(hf_result, "class_probs", {}) if hf_result is not None else {},
            "hf_error": getattr(hf_result, "error", None),
            **spectrum,
            **spectrogram,
            "harmonic_lines": harmonic_lines,
        },
    )


def apply_local_eval_guards(event: AcousticEvent) -> AcousticEvent:
    metadata = dict(event.metadata or {})
    harmonic = float(metadata.get("harmonic_evidence_pct_smoothed") or event.harmonic_evidence_pct or 0.0)
    raw_harmonic = float(metadata.get("harmonic_evidence_pct_raw") or event.harmonic_evidence_pct or 0.0)
    combined = float(event.combined_drone_evidence_pct or metadata.get("combined_drone_evidence_pct") or 0.0)
    ml = event.ml_drone_pct
    if ml is None:
        ml = metadata.get("ml_drone_pct")
    ml = None if ml is None else float(ml)
    hf_negative = event.hf_p_drone is not None and float(event.hf_p_drone) < 0.20
    status = event.status.value if hasattr(event.status, "value") else str(event.status)
    operator_label = event.operator_label or metadata.get("operator_label") or "background"

    if hf_negative and combined < 0.30 and status in {"alert", "drone_like"}:
        if max(harmonic, raw_harmonic) >= 0.45:
            status = "suspect"
            operator_label = "non_drone_harmonic"
            reason = "harmonic source detected, but ML strongly rejects drone"
        else:
            status = "background"
            operator_label = "background"
            reason = "ML strongly rejects drone and current combined evidence is weak"
        event.status = EventStatus(status)
        event.operator_label = operator_label
        event.decision_reason = reason
        metadata["operator_label"] = operator_label
        metadata["decision_reason"] = reason

    label_allowed = (
        (ml is not None and ml >= 0.90 and combined >= 0.45)
        or status == "alert"
        or int(event.channel_agreement_count or 0) >= 2
    )
    if operator_label == "drone_like" and not label_allowed:
        operator_label = "ml_drone_candidate" if ml is not None and ml >= 0.90 else "background"
        event.operator_label = operator_label
        metadata["operator_label"] = operator_label

    event.metadata = metadata
    return event


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
    parser.add_argument("--no-post", action="store_true", help="Build and print events without posting them to the server.")
    parser.add_argument("--reset-state-per-file", dest="reset_state_per_file", action="store_true", default=True)
    parser.add_argument("--no-reset-state-per-file", dest="reset_state_per_file", action="store_false")
    parser.add_argument("--skip-tail-padding", dest="skip_tail_padding", action="store_true", default=True)
    parser.add_argument("--keep-tail-padding", dest="skip_tail_padding", action="store_false")
    parser.add_argument("--min-rms", type=float, default=0.0)
    parser.add_argument("--eval-mode", action="store_true")
    parser.add_argument("--save-report", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    if args.eval_mode:
        args.reset_state_per_file = True
        args.skip_tail_padding = True
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
        if args.reset_state_per_file:
            detector_state = detector_state_from_args(args)
            last_hf_result = None
        files_seen += 1

        for window, padding_ratio in iter_audio_windows_with_padding(
            mono,
            int(args.sample_rate),
            float(args.window_sec),
            bool(args.skip_tail_padding),
        ):
            if args.max_windows is not None and windows_sent >= int(args.max_windows):
                if args.save_report:
                    write_report(args.save_report, events)
                return

            loop_start = time.time()
            window_rms = _window_rms(window)
            if float(args.min_rms or 0.0) > 0.0 and window_rms < float(args.min_rms):
                continue
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
            event.metadata["padding_ratio"] = padding_ratio
            event.metadata["window_rms"] = window_rms
            event = apply_local_eval_guards(event)
            if not args.no_post:
                requests.post(args.server, json=event.model_dump(mode="json"), timeout=2.0)
            events.append(event)
            print(format_detection_log(f"{path.name} {metadata.get('label')} {metadata.get('distance_category')}", event))
            windows_sent += 1
            if args.realtime:
                time.sleep(max(0.0, float(args.window_sec) - (time.time() - loop_start)))

    if args.save_report:
        write_report(args.save_report, events)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
