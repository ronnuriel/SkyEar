from __future__ import annotations

import argparse
import re
import time
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import requests
from scipy.signal import resample_poly

from shared.event_schema import AcousticEvent, EventStatus, GeoPoint
from station.audio_capture import to_mono
from station.detector_state import StationDetectorState, StationDetectorStateConfig
from station.hf_detector import DEFAULT_MODEL_ID, HFDetector
from station.spectrum import compute_harmonic_lines, compute_spectrogram_summary, compute_spectrum_summary


DEFAULT_DATASET_ID = "ahlab-drone-project/DroneAudioSet"
DEFAULT_DATASET_CONFIG = "drone-only"
DEFAULT_DATASET_SPLIT = "train_001"
DETECTOR_VERSION = "hf-dataset-stream-v1"


def mono_to_simulated_channels(mono: np.ndarray, channels: int) -> np.ndarray:
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    channels = int(channels)
    if channels <= 1:
        return mono.reshape(-1, 1)

    gains = np.linspace(1.0, 0.76, channels, dtype=np.float32)
    out = np.zeros((mono.size, channels), dtype=np.float32)
    for idx in range(channels):
        shift = idx % 5
        shifted = np.roll(mono, shift)
        if shift:
            shifted[:shift] = 0.0
        out[:, idx] = shifted * gains[idx]
    return out


def iter_audio_windows(mono: np.ndarray, sample_rate: int, window_sec: float) -> Iterator[np.ndarray]:
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    window_samples = max(1, int(round(float(sample_rate) * float(window_sec))))
    for start in range(0, mono.size, window_samples):
        window = mono[start : start + window_samples]
        if window.size < window_samples:
            window = np.pad(window, (0, window_samples - window.size))
        yield window.astype(np.float32)


def resample_mono(mono: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    source_sr = int(source_sr)
    target_sr = int(target_sr)
    if source_sr == target_sr:
        return mono

    try:
        import librosa

        return librosa.resample(mono, orig_sr=source_sr, target_sr=target_sr).astype(np.float32)
    except Exception:
        gcd = int(np.gcd(source_sr, target_sr))
        return resample_poly(mono, target_sr // gcd, source_sr // gcd).astype(np.float32)


def detector_config_from_args(args: argparse.Namespace) -> StationDetectorStateConfig:
    return StationDetectorStateConfig(
        f0_min=int(args.f0_min),
        f0_max=int(args.f0_max),
        max_freq=int(args.max_freq),
        min_suspect_threshold=float(args.suspect_threshold),
        min_alert_threshold=float(args.alert_threshold),
        calibration_seconds=float(args.calibration_seconds),
    )


def parse_file_path_metadata(file_path: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"dataset_file_path": file_path}
    if not file_path:
        return metadata

    parts = str(file_path).split("/")
    for part in parts:
        drone_match = re.fullmatch(r"(drone[^-/]+)(?:-.*)?", part)
        if drone_match:
            metadata["drone_id"] = drone_match.group(1)

        distance_match = re.search(r"mic-dist-([0-9.]+)(cm|m)", part)
        if distance_match:
            value = float(distance_match.group(1))
            unit = distance_match.group(2)
            metadata["distance_label"] = f"{distance_match.group(1)}{unit}"
            metadata["distance_m"] = value / 100.0 if unit == "cm" else value

        throttle_match = re.fullmatch(r"throttle-([^/]+)", part)
        if throttle_match:
            metadata["throttle"] = throttle_match.group(1)

    filename = parts[-1] if parts else ""
    mic_match = re.match(r"(mic\d+)_([^/]+?)(?:-File|\.)", filename)
    if mic_match:
        metadata["mic_id"] = mic_match.group(1)
        metadata["array_info"] = mic_match.group(2)

    return metadata


def find_audio_column(example: dict[str, Any]) -> str:
    if isinstance(example.get("audio"), dict) and "array" in example["audio"] and "sampling_rate" in example["audio"]:
        return "audio"
    for key, value in example.items():
        if isinstance(value, dict) and "array" in value and "sampling_rate" in value:
            return key
    raise ValueError("Could not find an audio column with array and sampling_rate.")


def extract_mono_audio(example: dict[str, Any], audio_column: str | None = None) -> tuple[np.ndarray, int, str]:
    audio_column = audio_column or find_audio_column(example)
    audio = example[audio_column]
    array = np.asarray(audio["array"], dtype=np.float32)
    if array.ndim == 2:
        array = to_mono(array)
    else:
        array = array.reshape(-1)
    return array.astype(np.float32), int(audio["sampling_rate"]), audio_column


def dataset_label(
    example: dict[str, Any],
    features: Any = None,
    audio_column: str | None = None,
    fallback: str = "unknown",
) -> str:
    candidates = ["label", "labels", "class", "category", "target", "data_type"]
    for key in candidates:
        if key not in example or key == audio_column:
            continue
        value = example[key]
        feature = None
        if features is not None:
            try:
                feature = features[key]
            except Exception:
                feature = None
        if feature is not None and hasattr(feature, "int2str"):
            try:
                return str(feature.int2str(value))
            except Exception:
                pass
        return str(value)
    return fallback


def label_matches(label: str, label_filter: str | None) -> bool:
    if not label_filter:
        return True
    return label_filter.lower() in str(label).lower()


def build_event(
    *,
    station_id: str,
    dataset_id: str,
    dataset_config: str | None,
    dataset_idx: int,
    dataset_label_value: str,
    audio: np.ndarray,
    sample_rate: int,
    timestamp: float,
    detector_state: StationDetectorState,
    hf_result: Any = None,
    hf_p_drone: float | None = None,
    file_metadata: dict[str, Any] | None = None,
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
    file_metadata = file_metadata or {}

    return AcousticEvent(
        station_id=station_id,
        station_name=f"HF Dataset {dataset_id}",
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
            "source": "huggingface_dataset",
            "dataset_id": dataset_id,
            "dataset_config": dataset_config,
            "dataset_label": dataset_label_value,
            "dataset_index": dataset_idx,
            "sample_rate": sample_rate,
            "channels": frame.channel_count,
            "mic_profile": "simulated_multichannel_from_dataset",
            "mic_sync_mode": "unsynchronized",
            "suspect_threshold": frame.suspect_threshold,
            "alert_threshold": frame.alert_threshold,
            "f0_stable": frame.f0_stable,
            "hf_label": getattr(hf_result, "label", None),
            "hf_class_probs": getattr(hf_result, "class_probs", {}) if hf_result is not None else {},
            "hf_error": getattr(hf_result, "error", None),
            **file_metadata,
            **spectrum,
            **spectrogram,
            "harmonic_lines": harmonic_lines,
        },
    )


def load_dataset_iterable(
    dataset_id: str,
    split: str,
    streaming: bool,
    config_name: str | None = None,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required; install requirements_hf.txt") from exc
    return load_dataset(dataset_id, name=config_name, split=split, streaming=streaming)


def resolve_dataset_options(dataset_id: str, config_name: str | None, split: str) -> tuple[str | None, str]:
    if dataset_id != DEFAULT_DATASET_ID:
        return config_name, split

    resolved_config = config_name or DEFAULT_DATASET_CONFIG
    resolved_split = DEFAULT_DATASET_SPLIT if split == "train" else split
    return resolved_config, resolved_split


def calibrate_with_background(args: argparse.Namespace, detector_state: StationDetectorState) -> None:
    if not args.background_dataset:
        return

    background_config, background_split = resolve_dataset_options(
        args.background_dataset,
        args.background_config,
        args.background_split,
    )
    dataset = load_dataset_iterable(
        args.background_dataset,
        background_split,
        args.streaming,
        config_name=background_config,
    )
    audio_column = None
    samples_seen = 0
    timestamp = time.time()

    for example in dataset:
        if samples_seen >= int(args.background_samples):
            break
        if audio_column is None:
            audio_column = find_audio_column(example)

        mono, source_sr, _ = extract_mono_audio(example, audio_column)
        mono = resample_mono(mono, source_sr, int(args.sample_rate))
        samples_seen += 1

        for window in iter_audio_windows(mono, int(args.sample_rate), float(args.window_sec)):
            audio = mono_to_simulated_channels(window, int(args.channels))
            detector_state.update(audio, int(args.sample_rate), timestamp)
            timestamp += float(args.window_sec)

    print(
        "background calibration "
        f"samples={samples_seen} calibrated={detector_state.calibrated} "
        f"th={detector_state.suspect_threshold:.1f}/{detector_state.alert_threshold:.1f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream Hugging Face drone-audio dataset examples into SkyEar as station events. "
            "Raw audio is processed locally and is not sent to the server."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--config",
        default=None,
        help=f"Dataset config/name. Defaults to {DEFAULT_DATASET_CONFIG!r} for {DEFAULT_DATASET_ID}.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--station-id", default="hf_dataset_001")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-windows", type=int, default=None, help="Optional cap on total posted audio windows.")
    parser.add_argument("--calibration-seconds", type=float, default=0.0)
    parser.add_argument("--suspect-threshold", type=float, default=16.0)
    parser.add_argument("--alert-threshold", type=float, default=22.0)
    parser.add_argument("--f0-min", type=int, default=500)
    parser.add_argument("--f0-max", type=int, default=2200)
    parser.add_argument("--max-freq", type=int, default=7000)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--label-filter")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--background-dataset")
    parser.add_argument("--background-config")
    parser.add_argument("--background-split", default="train")
    parser.add_argument("--background-samples", type=int, default=20)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    dataset_config, dataset_split = resolve_dataset_options(args.dataset, args.config, args.split)

    dataset = load_dataset_iterable(args.dataset, dataset_split, args.streaming, config_name=dataset_config)
    features = getattr(dataset, "features", None)
    detector_config = detector_config_from_args(args)
    detector_state = StationDetectorState(detector_config)
    if args.background_dataset:
        calibrate_with_background(args, detector_state)
    elif detector_config.calibration_seconds <= 0.0:
        detector_state.calibrated = True

    hf_detector = HFDetector(model_id=args.model_id) if args.hf else None
    last_hf_result = None
    audio_column = None
    windows_sent = 0
    samples_seen = 0

    for dataset_idx, example in enumerate(dataset):
        if samples_seen >= int(args.max_samples):
            break

        if audio_column is None:
            audio_column = find_audio_column(example)
        label = dataset_label(
            example,
            features=features,
            audio_column=audio_column,
            fallback=dataset_config or "unknown",
        )
        if not label_matches(label, args.label_filter):
            continue

        mono, source_sr, _ = extract_mono_audio(example, audio_column)
        mono = resample_mono(mono, source_sr, int(args.sample_rate))
        file_path = example.get("file_path")
        if not file_path and isinstance(example.get(audio_column), dict):
            file_path = example[audio_column].get("path")
        file_metadata = parse_file_path_metadata(file_path)
        samples_seen += 1

        for window in iter_audio_windows(mono, int(args.sample_rate), float(args.window_sec)):
            if args.max_windows is not None and windows_sent >= int(args.max_windows):
                return
            loop_start = time.time()
            audio = mono_to_simulated_channels(window, int(args.channels))
            if hf_detector is not None and windows_sent % 2 == 0:
                last_hf_result = hf_detector.predict(window, int(args.sample_rate))
            hf_p_drone = last_hf_result.p_drone if last_hf_result is not None else None
            event = build_event(
                station_id=args.station_id,
                dataset_id=args.dataset,
                dataset_config=dataset_config,
                dataset_idx=dataset_idx,
                dataset_label_value=label,
                audio=audio,
                sample_rate=int(args.sample_rate),
                timestamp=loop_start,
                detector_state=detector_state,
                hf_result=last_hf_result,
                hf_p_drone=hf_p_drone,
                file_metadata=file_metadata,
            )
            distance = file_metadata.get("distance_m")
            distance_display = f"{distance:g}m" if distance is not None else "None"
            requests.post(args.server, json=event.model_dump(mode="json"), timeout=2.0)
            print(
                f"{dataset_idx} {label} {event.status:11s} "
                f"hf_p={event.hf_p_drone} harm={event.harmonic_score:.1f} "
                f"f0={event.best_f0_hz} dist={distance_display} "
                f"throttle={file_metadata.get('throttle')} drone={file_metadata.get('drone_id')}"
            )
            windows_sent += 1
            if args.realtime:
                time.sleep(max(0.0, float(args.window_sec) - (time.time() - loop_start)))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
