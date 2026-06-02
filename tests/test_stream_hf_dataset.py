from __future__ import annotations

import sys

import numpy as np

from station.detector_state import StationDetectorState, StationDetectorStateConfig
from tools.stream_hf_dataset import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_SPLIT,
    build_event,
    dataset_label,
    iter_audio_windows,
    mono_to_simulated_channels,
    parse_args,
    parse_file_path_metadata,
    resolve_dataset_options,
)


def _metadata_has_key(metadata, forbidden: set[str]) -> bool:
    if isinstance(metadata, dict):
        return any(key in forbidden or _metadata_has_key(value, forbidden) for key, value in metadata.items())
    if isinstance(metadata, list):
        return any(_metadata_has_key(value, forbidden) for value in metadata)
    return False


def test_mono_to_simulated_channels_returns_samples_by_channels():
    mono = np.linspace(-1.0, 1.0, 12, dtype=np.float32)

    channels = mono_to_simulated_channels(mono, 4)

    assert channels.shape == (12, 4)
    assert channels.dtype == np.float32
    assert np.allclose(channels[:, 0], mono)


def test_iter_audio_windows_pads_final_window():
    mono = np.arange(10, dtype=np.float32)

    windows = list(iter_audio_windows(mono, sample_rate=10, window_sec=0.3))

    assert len(windows) == 4
    assert all(window.shape == (3,) for window in windows)
    assert np.allclose(windows[0], [0, 1, 2])
    assert np.allclose(windows[-1], [9, 0, 0])


def test_primary_dataset_defaults_to_existing_config_and_split():
    config, split = resolve_dataset_options(DEFAULT_DATASET_ID, None, "train")

    assert config == DEFAULT_DATASET_CONFIG
    assert split == DEFAULT_DATASET_SPLIT


def test_dataset_label_uses_data_type_column():
    assert dataset_label({"data_type": "drone-only"}, fallback="fallback") == "drone-only"


def test_parse_file_path_metadata_extracts_dataset_context():
    metadata = parse_file_path_metadata(
        "drone-only-recordings/drone2-only/mic-dist-50cm/throttle-low/mic2_8array-down-File3.wav"
    )

    assert metadata["distance_m"] == 0.5
    assert metadata["distance_label"] == "50cm"
    assert metadata["throttle"] == "low"
    assert metadata["drone_id"] == "drone2"
    assert metadata["mic_id"] == "mic2"
    assert metadata["array_info"] == "8array-down"


def test_parse_args_calibration_seconds_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stream_hf_dataset.py"])

    args = parse_args()

    assert args.calibration_seconds == 0.0


def test_event_metadata_contains_dataset_context_without_raw_audio():
    sample_rate = 44100
    mono = np.zeros(sample_rate, dtype=np.float32)
    audio = mono_to_simulated_channels(mono, 2)
    detector_state = StationDetectorState(StationDetectorStateConfig(calibration_seconds=0.0))

    event = build_event(
        station_id="hf_dataset_test",
        dataset_id="example/drone-data",
        dataset_config="drone-only",
        dataset_idx=7,
        dataset_label_value="drone",
        audio=audio,
        sample_rate=sample_rate,
        timestamp=1.0,
        detector_state=detector_state,
        file_metadata=parse_file_path_metadata(
            "drone-only-recordings/drone2-only/mic-dist-50cm/throttle-low/mic2_8array-down-File3.wav"
        ),
    )

    assert event.metadata["source"] == "huggingface_dataset"
    assert event.metadata["dataset_id"] == "example/drone-data"
    assert event.metadata["dataset_config"] == "drone-only"
    assert event.metadata["dataset_label"] == "drone"
    assert event.metadata["dataset_index"] == 7
    assert event.metadata["dataset_file_path"].endswith("mic2_8array-down-File3.wav")
    assert event.metadata["distance_m"] == 0.5
    assert event.metadata["throttle"] == "low"
    assert event.metadata["drone_id"] == "drone2"
    assert event.metadata["mic_id"] == "mic2"
    assert event.metadata["array_info"] == "8array-down"
    assert event.metadata["mic_sync_mode"] == "unsynchronized"
    assert event.station_mode == "unsynchronized_multimic_voting"
    assert not _metadata_has_key(event.metadata, {"audio", "array", "raw_audio", "waveform"})


def test_event_metadata_marks_hf_negative_for_high_harmonic_source():
    sample_rate = 44100
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = np.zeros_like(t)
    for k in range(1, 5):
        mono += (0.05 / k) * np.sin(2 * np.pi * 700 * k * t)
    audio = mono_to_simulated_channels(mono.astype(np.float32), 1)
    detector_state = StationDetectorState(StationDetectorStateConfig(calibration_seconds=0.0))
    detector_state.calibrated = True

    event = build_event(
        station_id="hf_dataset_test",
        dataset_id="example/source-data",
        dataset_config="source-only",
        dataset_idx=3,
        dataset_label_value="source-only",
        audio=audio,
        sample_rate=sample_rate,
        timestamp=1.0,
        detector_state=detector_state,
        hf_p_drone=0.001,
    )

    assert event.status == "suspect"
    assert event.hf_negative is True
    assert event.metadata["hf_negative"] is True
    assert event.metadata["harmonic_evidence_pct"] > 0.75
    assert event.metadata["ml_drone_pct"] == 0.001
    assert event.metadata["decision_reason"] == "harmonic source detected, but ML strongly rejects drone"
