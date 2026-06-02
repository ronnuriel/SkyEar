from __future__ import annotations

import numpy as np

from station.detector_state import StationDetectorState, StationDetectorStateConfig
from tools.stream_hf_dataset import build_event, iter_audio_windows, mono_to_simulated_channels


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


def test_event_metadata_contains_dataset_context_without_raw_audio():
    sample_rate = 44100
    mono = np.zeros(sample_rate, dtype=np.float32)
    audio = mono_to_simulated_channels(mono, 2)
    detector_state = StationDetectorState(StationDetectorStateConfig(calibration_seconds=0.0))

    event = build_event(
        station_id="hf_dataset_test",
        dataset_id="example/drone-data",
        dataset_idx=7,
        dataset_label_value="drone",
        audio=audio,
        sample_rate=sample_rate,
        timestamp=1.0,
        detector_state=detector_state,
    )

    assert event.metadata["source"] == "huggingface_dataset"
    assert event.metadata["dataset_id"] == "example/drone-data"
    assert event.metadata["dataset_label"] == "drone"
    assert event.metadata["dataset_index"] == 7
    assert event.metadata["mic_sync_mode"] == "unsynchronized"
    assert event.station_mode == "unsynchronized_multimic_voting"
    assert not _metadata_has_key(event.metadata, {"audio", "array", "raw_audio", "waveform"})
