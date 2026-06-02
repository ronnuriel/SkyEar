from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from tools.stream_hf_dataset import mono_to_simulated_channels
from tools.stream_hf_dataset import format_detection_log
from tools.stream_local_dataset import (
    apply_local_eval_guards,
    build_event,
    detector_state_from_args,
    infer_path_metadata,
    iter_audio_windows_with_padding,
    metadata_matches,
    read_audio_mono,
)


class _Args:
    reset_state_per_file = True
    skip_tail_padding = True
    eval_mode = False


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
    assert metadata["distance_category"] == "distant"


def test_infer_path_metadata_maps_far_to_distant():
    metadata = infer_path_metadata("Far/background/example.wav")

    assert metadata["label"] == "background"
    assert metadata["distance_category"] == "distant"


def test_metadata_filter_matching_is_case_insensitive():
    metadata = {"label": "drone", "distance_category": "distant"}

    assert metadata_matches(metadata, "DRONE", "Distant") is True
    assert metadata_matches(metadata, "bird", "Distant") is False


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
    assert event.metadata["distance_category"] == "close"
    assert event.metadata["file_path"].endswith("sample.wav")
    assert event.channel_count == 1
    assert not _metadata_has_key(event.metadata, {"audio", "array", "raw_audio", "waveform"})


def test_detection_log_contains_operator_and_score_details(tmp_path):
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    wav_path = tmp_path / "Distant" / "drone" / "BACKGROUND_001.wav"
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
        hf_p_drone=0.001,
    )

    line = format_detection_log("BACKGROUND_001.wav background", event)

    assert "BACKGROUND_001.wav background" in line
    assert "label=" in line
    assert "hf=0.001" in line
    assert "harm=" in line
    assert "h=" in line
    assert "combined=" in line
    assert "canon=" in line
    assert "stable=" in line
    assert "reason=\"" in line


def test_reset_state_per_file_prevents_alert_inheritance():
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    harmonic = (0.1 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    silence = np.zeros(sample_rate, dtype=np.float32)

    first_state = detector_state_from_args(_Args())
    for idx in range(3):
        build_event(
            station_id="local_test",
            root=".",
            file_path="drone.wav",
            label="drone",
            distance_category="close",
            audio=mono_to_simulated_channels(harmonic, 1),
            sample_rate=sample_rate,
            timestamp=float(idx),
            detector_state=first_state,
            hf_p_drone=0.99,
        )

    second_state = detector_state_from_args(_Args())
    event = build_event(
        station_id="local_test",
        root=".",
        file_path="background.wav",
        label="background",
        distance_category="distant",
        audio=mono_to_simulated_channels(silence, 1),
        sample_rate=sample_rate,
        timestamp=10.0,
        detector_state=second_state,
        hf_p_drone=0.001,
    )

    assert event.status.value != "alert"
    assert event.status.value != "drone_like"


def test_tail_padding_window_is_skipped_when_mostly_padding():
    sample_rate = 100
    mono = np.ones(250, dtype=np.float32)

    windows = list(iter_audio_windows_with_padding(mono, sample_rate, 1.0, skip_tail_padding=True))

    assert len(windows) == 2
    assert all(padding_ratio == 0.0 for _, padding_ratio in windows)


def test_background_hf_high_combined_zero_is_candidate_at_most():
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32)
    state = detector_state_from_args(_Args())

    event = build_event(
        station_id="local_test",
        root=".",
        file_path="background.wav",
        label="background",
        distance_category="distant",
        audio=mono_to_simulated_channels(silence, 1),
        sample_rate=sample_rate,
        timestamp=1.0,
        detector_state=state,
        hf_p_drone=0.95,
    )
    event = apply_local_eval_guards(event)

    assert event.combined_drone_evidence_pct == 0.0
    assert event.status.value != "alert"
    assert event.operator_label in {"ml_drone_candidate", "background"}
    assert event.operator_label != "drone_like"


def test_hf_low_high_harmonic_is_non_drone_harmonic_not_alert():
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    harmonic = (0.1 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    state = detector_state_from_args(_Args())

    event = None
    for idx in range(3):
        event = build_event(
            station_id="local_test",
            root=".",
            file_path="helicopter.wav",
            label="helicopter",
            distance_category="close",
            audio=mono_to_simulated_channels(harmonic, 1),
            sample_rate=sample_rate,
            timestamp=float(idx),
            detector_state=state,
            hf_p_drone=0.001,
        )
    event = apply_local_eval_guards(event)

    assert event is not None
    assert event.status.value != "alert"
    assert event.operator_label == "non_drone_harmonic"
