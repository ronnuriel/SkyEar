from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from dashboard.local_station_app import load_history_rows, parse_snapshot, snapshot_is_stale
from shared.event_schema import AcousticEvent, EventStatus
from station.local_monitor import (
    atomic_write_json,
    build_local_monitor_snapshot,
    decimated_waveform,
    history_row_from_event,
    write_local_monitor_snapshot,
)


def _event() -> AcousticEvent:
    return AcousticEvent(
        station_id="station_001",
        station_name="Local Station",
        timestamp_unix=100.0,
        status=EventStatus.SUSPECT,
        confidence=0.7,
        harmonic_score=18.0,
        harmonic_evidence_pct_smoothed=0.4,
        ml_drone_pct=0.8,
        combined_drone_evidence_pct=0.53,
        hf_p_drone=0.8,
        operator_label="ml_drone_candidate",
        candidate_run=2,
        ml_positive_run=2,
        strong_run=0,
        best_f0_hz=920,
        two_mic_side="left",
        two_mic_delay_us=1500.0,
        two_mic_angle_from_center_deg=15.0,
        two_mic_confidence=0.62,
        two_mic_peak_ratio=1.8,
        two_mic_look_label="left",
        two_mic_look_hint="LOOK LEFT - approx 15 deg from center, scan +/-30 deg, front/back ambiguous",
        two_mic_sector_width_deg=60.0,
        two_mic_front_back_ambiguous=True,
        two_mic_direction_stable=True,
        rms=0.03,
        peak=0.2,
        metadata={
            "sample_rate": 44100,
            "hf_label": "drone",
            "calibration_loaded": True,
            "calibration_file": "configs/array_calibration_station_001.json",
            "channel_rms": [0.1, 0.2],
            "channel_health": ["ok", "dropout"],
            "bad_channels": [1],
            "two_mic_direction_enabled": True,
            "two_mic_stable_window_count": 3,
            "two_mic_tracker_window_count": 5,
        },
    )


def test_decimated_waveform_respects_max_length():
    waveform = decimated_waveform(np.arange(5000, dtype=np.float32), max_points=1200)

    assert len(waveform) == 1200
    assert waveform[0] == 0.0


def test_latest_json_write_is_atomic_and_readable(tmp_path: Path):
    path = tmp_path / "station_latest.json"

    atomic_write_json(path, {"ok": True, "value": 3})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True, "value": 3}
    assert not list(tmp_path.glob("*.tmp"))


def test_local_monitor_payload_contains_event_audio_spectrum_spectrogram(tmp_path: Path):
    event = _event()
    mono = np.sin(np.linspace(0, 1, 2048)).astype(np.float32)
    spectrum = {"spectrum_freqs_hz": [0.0, 100.0], "spectrum_db": [-20.0, 0.0]}
    spectrogram = {
        "spectrogram_freqs_hz": [0.0, 100.0],
        "spectrogram_times_sec": [0.0, 0.5],
        "spectrogram_db": [[-20.0, -10.0], [-5.0, 0.0]],
    }
    snapshot = build_local_monitor_snapshot(
        event=event,
        mono=mono,
        waveform_points=32,
        spectrum=spectrum,
        spectrogram=spectrogram,
        harmonic_lines=[{"k": 1, "freq_hz": 920.0}],
        beam_result=SimpleNamespace(beam_scan_deg=[0.0, 5.0], beam_scan_score=[0.1, 0.2]),
        server_state={"last_send_error": "server down"},
        updated_unix=101.0,
    )

    assert snapshot["event"]["station_id"] == "station_001"
    assert len(snapshot["audio"]["waveform"]) == 32
    assert snapshot["spectrum"] == spectrum
    assert snapshot["spectrogram"] == spectrogram
    assert snapshot["beam"]["beam_scan_deg"] == [0.0, 5.0]
    assert snapshot["beam"]["two_mic_look_label"] == "left"
    assert snapshot["beam"]["two_mic_angle_from_center_deg"] == 15.0
    assert snapshot["beam"]["two_mic_direction_stable"] is True
    assert snapshot["beam"]["two_mic_stable_window_count"] == 3
    assert snapshot["audio"]["calibration_loaded"] is True
    assert snapshot["audio"]["channel_health"] == ["ok", "dropout"]
    assert "raw_audio" not in snapshot["audio"]

    state_path = tmp_path / "station_latest.json"
    history_path = tmp_path / "station_history.jsonl"
    write_local_monitor_snapshot(
        state_path=state_path,
        history_path=history_path,
        snapshot=snapshot,
        history_row=history_row_from_event(event),
    )
    parsed = parse_snapshot(json.loads(state_path.read_text(encoding="utf-8")))
    assert parsed["station_id"] == "station_001"
    assert load_history_rows(history_path)[0]["operator_label"] == "ml_drone_candidate"


def test_stale_state_helper_returns_true_when_updated_unix_is_old():
    assert snapshot_is_stale({"updated_unix": 100.0}, stale_after_sec=3.0, now=104.0)
    assert not snapshot_is_stale({"updated_unix": 100.0}, stale_after_sec=3.0, now=102.0)


def test_local_station_app_parsing_works_with_minimal_snapshot():
    parsed = parse_snapshot(
        {
            "updated_unix": 100.0,
            "event": {"station_id": "s1", "status": "background"},
            "audio": {"waveform": [0.0]},
        }
    )

    assert parsed["station_id"] == "s1"
    assert parsed["spectrum"] == {}
    assert parsed["spectrogram"] == {}
    assert parsed["hf"] == {}
