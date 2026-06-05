from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from station.audio_capture import CapturedAudioBlock, ThreadedAudioCapture
from station.recording_manager import RecordingManager
from station.station_agent import _finalize_station_recording
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


def test_recording_manager_stop_flushes_short_partial_chunk(tmp_path: Path):
    manager = _manager(tmp_path, channels=1, chunk_sec=60)

    manager.start_recording("short_partial")
    manager.append_audio(np.ones((200, 1), dtype=np.float32), timestamp=100.0)
    state = manager.stop_recording()

    wavs = sorted(Path(state["session_dir"]).glob("*.wav"))
    assert len(wavs) == 1
    sr, data = wavfile.read(wavs[0])
    assert sr == 8000
    assert data.shape[0] == 200


def test_recording_finalizer_flushes_active_recording_once(tmp_path: Path):
    manager = _manager(tmp_path, channels=1, chunk_sec=60)

    manager.start_recording("finalizer")
    manager.append_audio(np.ones((100, 1), dtype=np.float32), timestamp=100.0)
    first = _finalize_station_recording(manager)
    second = _finalize_station_recording(manager)

    wavs = sorted(Path(first["session_dir"]).glob("*.wav"))
    assert len(wavs) == 1
    assert first["recording"] is False
    assert second["recording"] is False
    assert len(second["wav_files"]) == 1


def test_recording_manager_stop_without_audio_keeps_metadata_but_no_wav(tmp_path: Path):
    manager = _manager(tmp_path, channels=1, chunk_sec=60)

    manager.start_recording("empty")
    state = manager.stop_recording()
    again = manager.stop_recording()

    session_dir = Path(state["session_dir"])
    assert (session_dir / "metadata.json").exists()
    assert (session_dir / "markers.csv").exists()
    assert sorted(session_dir.glob("*.wav")) == []
    assert again["recording"] is False


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


def test_start_label_writes_initial_marker_and_manual_mark_before_wav_flush(tmp_path: Path):
    manager = _manager(tmp_path, channels=1, chunk_sec=60)

    manager.start_recording("marker_start", label="hover")
    manager.mark_event("hover", note="manual mark before flush")
    state = manager.stop_recording()

    markers = Path(state["session_dir"]) / "markers.csv"
    rows = list(csv.DictReader(markers.open("r", newline="", encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["source"] == "start"
    assert rows[0]["label"] == "hover"
    assert rows[0]["offset_sec"] == "0.000"
    assert rows[0]["chunk_index_expected"] == "0"
    assert rows[1]["source"] == "manual"
    assert rows[1]["note"] == "manual mark before flush"
    assert rows[1]["current_wav_path"] == ""
    assert state["marker_count"] == 2


def test_recording_metadata_uses_capture_timestamps_and_records_gaps(tmp_path: Path):
    manager = _manager(tmp_path, channels=1, chunk_sec=60)

    manager.start_recording("gaps")
    manager.append_audio(np.ones((800, 1), dtype=np.float32), timestamp=100.0)
    manager.append_audio(np.ones((800, 1), dtype=np.float32), timestamp=101.0)
    state = manager.stop_recording()

    metadata = json.loads((Path(state["session_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    wav = metadata["wav_files"][0]
    assert wav["start_time"] == 100.0
    assert abs(wav["end_time"] - 101.1) < 1e-6
    assert abs(wav["duration_sec"] - 0.2) < 1e-6
    assert metadata["discontinuities"]
    assert metadata["discontinuities"][0]["missing_sec"] > 0.8


def test_threaded_capture_keeps_recording_continuous_when_detection_lags(tmp_path: Path):
    sample_rate = 100
    block_samples = 10
    total_blocks = 30
    manager = RecordingManager(
        station_id="station_test",
        sample_rate=sample_rate,
        channels=1,
        config={"enabled": True, "root": str(tmp_path / "recordings"), "chunk_sec": 60, "max_disk_gb": 20},
    )
    manager.start_recording("slow_detector")

    def source():
        for idx in range(total_blocks):
            start = 100.0 + idx * (block_samples / sample_rate)
            audio = np.ones((block_samples, 1), dtype=np.float32)
            yield CapturedAudioBlock(
                audio=audio,
                start_unix=start,
                end_unix=start + block_samples / sample_rate,
                input_overflow=False,
            )

    capture = ThreadedAudioCapture(
        device_id=None,
        sample_rate=sample_rate,
        channels=1,
        window_sec=block_samples / sample_rate,
        queue_size=2,
        on_block=lambda block: manager.append_audio(block.audio, timestamp=block.start_unix),
        source=source,
    )
    capture.start()
    processed = 0
    for _block in capture.blocks():
        processed += 1
        time.sleep(0.002)
    capture.stop()
    state = manager.stop_recording()

    wav_duration = sum(float(item["duration_sec"]) for item in state["wav_files"])
    assert abs(wav_duration - 3.0) < 1e-6
    assert state["recording_blocks_written"] == total_blocks
    assert capture.stats()["detection_blocks_dropped"] > 0
    assert processed < total_blocks


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
