from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from tools.stream_hf_dataset import mono_to_simulated_channels
from tools.stream_local_dataset import (
    build_event,
    detector_state_from_args,
    infer_path_metadata,
    read_audio_mono,
)


class _Args:
    pass


def _metadata_has_key(metadata, forbidden: set[str]) -> bool:
    if isinstance(metadata, dict):
        return any(key in forbidden or _metadata_has_key(value, forbidden) for key, value in metadata.items())
    if isinstance(metadata, list):
        return any(_metadata_has_key(value, forbidden) for value in metadata)
    return False


def test_infer_path_metadata_extracts_label_and_distance():
    root = "/datasets/svanstrom"
    metadata = infer_path_metadata(
        "/datasets/svanstrom/Distant/drone/session_01/example.wav",
        root=root,
    )

    assert metadata["label"] == "drone"
    assert metadata["distance_category"] == "Distant"


def test_infer_path_metadata_maps_far_to_distant():
    metadata = infer_path_metadata("Far/background/example.wav")

    assert metadata["label"] == "background"
    assert metadata["distance_category"] == "Distant"


def test_local_wav_event_builds_correctly_without_raw_audio_metadata(tmp_path):
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    wav_path = tmp_path / "Close" / "drone" / "sample.wav"
    wav_path.parent.mkdir(parents=True)
    wavfile.write(wav_path, sample_rate, audio)

    mono, source_sr = read_audio_mono(wav_path)
    channels = mono_to_simulated_channels(mono, 1)
    detector_state = detector_state_from_args(_Args())
    metadata = infer_path_metadata(wav_path, root=tmp_path)

    event = build_event(
        station_id="local_test",
        root=tmp_path,
        file_path=wav_path,
        label=metadata["label"],
        distance_category=metadata["distance_category"],
        audio=channels,
        sample_rate=source_sr,
        timestamp=1.0,
        detector_state=detector_state,
    )

    assert event.station_id == "local_test"
    assert event.metadata["source"] == "local_distance_dataset"
    assert event.metadata["label"] == "drone"
    assert event.metadata["distance_category"] == "Close"
    assert event.metadata["file_path"].endswith("sample.wav")
    assert event.channel_count == 1
    assert not _metadata_has_key(event.metadata, {"audio", "array", "raw_audio", "waveform"})
