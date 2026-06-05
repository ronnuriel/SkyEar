from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from station.recording_manager import RecordingManager
from tools.build_recording_manifest import build_manifest


def _manager(tmp_path: Path, channels: int = 1, **cfg) -> RecordingManager:
    values = {
        "enabled": True,
        "root": str(tmp_path / "recordings"),
        "chunk_sec": 0.1,
        "max_session_sec": 3600,
        "max_disk_gb": 20,
    }
    values.update(cfg)
    return RecordingManager(
        station_id="station_test",
        sample_rate=8000,
        channels=channels,
        config=values,
        station_config={"station": {"station_id": "station_test"}},
    )


def test_recording_manager_writes_mono_wav_chunks(tmp_path: Path):
    manager = _manager(tmp_path, channels=1)

    manager.start_recording("mono_test")
    manager.append_audio(np.zeros((1200, 1), dtype=np.float32), timestamp=100.0)
    state = manager.stop_recording()

    wavs = sorted(Path(state["session_dir"]).glob("*.wav"))
    assert wavs
    sr, data = wavfile.read(wavs[0])
    assert sr == 8000
    assert data.ndim == 1


def test_recording_manager_writes_8ch_wav_chunks(tmp_path: Path):
    manager = _manager(tmp_path, channels=8)

    manager.start_recording("array_test")
    manager.append_audio(np.zeros((1200, 8), dtype=np.float32), timestamp=100.0)
    state = manager.stop_recording()

    wavs = sorted(Path(state["session_dir"]).glob("*.wav"))
    assert wavs
    sr, data = wavfile.read(wavs[0])
    assert sr == 8000
    assert data.ndim == 2
    assert data.shape[1] == 8


def test_recording_manager_start_stop_state_transitions(tmp_path: Path):
    manager = _manager(tmp_path)

    started = manager.start_recording("state_test")
    stopped = manager.stop_recording()

    assert started["recording"] is True
    assert stopped["recording"] is False
    assert stopped["session_dir"]


def test_recording_manager_marker_csv_written(tmp_path: Path):
    manager = _manager(tmp_path)

    manager.start_recording("markers")
    manager.mark_event("hover", note="DJI Neo 20m", distance_m=20.0, bearing_deg=45.0, drone_model="DJI Neo")
    state = manager.stop_recording()

    markers = Path(state["session_dir"]) / "markers.csv"
    rows = list(csv.DictReader(markers.open("r", newline="", encoding="utf-8")))
    assert rows[0]["label"] == "hover"
    assert rows[0]["note"] == "DJI Neo 20m"
    assert rows[0]["distance_m"] == "20.000"
    assert rows[0]["bearing_deg"] == "45.000"
    assert rows[0]["drone_model"] == "DJI Neo"


def test_recording_manager_disk_limit_prevents_new_recording(tmp_path: Path):
    root = tmp_path / "recordings"
    root.mkdir()
    (root / "existing.bin").write_bytes(b"xx")
    manager = _manager(tmp_path, root=str(root), max_disk_gb=1e-12)

    state = manager.start_recording("blocked")

    assert state["recording"] is False
    assert state["last_error"] == "recording_disk_limit_reached"


def test_recording_manifest_finds_recorded_wavs(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.start_recording("manifest")
    manager.append_audio(np.zeros((1200, 1), dtype=np.float32), timestamp=100.0)
    manager.mark_event("hover", note="training sample", distance_m=12.0, drone_model="DJI Neo")
    manager.stop_recording()
    output = tmp_path / "manifest.csv"

    rows = build_manifest(tmp_path / "recordings", output)

    assert output.exists()
    assert rows
    assert rows[0]["wav_path"].endswith(".wav")
    assert rows[0]["station_id"] == "station_test"
    assert rows[0]["label_from_markers"] == "hover"
