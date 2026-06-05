from __future__ import annotations

import numpy as np
import requests

from server.api import ingest_heartbeat, recording_mark, recording_start, recording_state, recording_stop
from server.database import db
from shared.event_schema import StationHeartbeat
from station.recording_manager import RecordingManager
from station.recording_control import RecordingControlServer
from station.station_agent import _execute_recording_command


def setup_function():
    db.events.clear()
    db.alerts.clear()
    db.heartbeats.clear()
    db.recording_commands.clear()


def test_server_recording_endpoints_queue_commands():
    start = recording_start("station_1")
    mark = recording_mark("station_1")
    stop = recording_stop("station_1")

    assert start["command"]["action"] == "start"
    assert mark["command"]["action"] == "mark"
    assert stop["command"]["action"] == "stop"
    assert len(db.pending_recording_commands("station_1")) == 3


def test_heartbeat_polls_one_recording_command():
    recording_start("station_1")
    heartbeat = StationHeartbeat(station_id="station_1", timestamp_unix=1.0)

    response = ingest_heartbeat(heartbeat)

    assert response["recording_command"]["action"] == "start"
    assert db.pending_recording_commands("station_1") == []


def test_recording_state_reads_heartbeat_metadata():
    heartbeat = StationHeartbeat(
        station_id="station_1",
        timestamp_unix=1.0,
        metadata={"recording_state": {"recording": True, "session_dir": "runtime/recordings/session"}},
    )
    ingest_heartbeat(heartbeat)

    state = recording_state("station_1")

    assert state["state"]["recording"] is True
    assert state["state"]["session_dir"] == "runtime/recordings/session"


def test_station_command_execution_start_stop_mark(tmp_path):
    manager = RecordingManager(
        station_id="station_1",
        sample_rate=8000,
        channels=1,
        config={"root": str(tmp_path), "chunk_sec": 0.1, "max_disk_gb": 20},
    )

    start_result = _execute_recording_command(
        manager,
        {"command_id": "1", "action": "start", "payload": {"session_name": "cmd_test"}},
    )
    manager.append_audio(np.zeros((800, 1), dtype=np.float32), timestamp=10.0)
    mark_result = _execute_recording_command(
        manager,
        {"command_id": "2", "action": "mark", "payload": {"label": "hover", "note": "cmd mark"}},
    )
    stop_result = _execute_recording_command(manager, {"command_id": "3", "action": "stop", "payload": {}})

    assert start_result["ok"] is True
    assert mark_result["ok"] is True
    assert stop_result["ok"] is True
    assert stop_result["state"]["recording"] is False


def test_local_recording_control_server_start_stop(tmp_path):
    manager = RecordingManager(
        station_id="station_1",
        sample_rate=8000,
        channels=1,
        config={"root": str(tmp_path), "chunk_sec": 0.1, "max_disk_gb": 20},
    )
    server = RecordingControlServer(manager, port=0)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        start = requests.post(f"{base}/recording/start", json={"session_name": "local"}, timeout=2).json()
        state = requests.get(f"{base}/recording/state", timeout=2).json()
        stop = requests.post(f"{base}/recording/stop", json={}, timeout=2).json()
    finally:
        server.stop()

    assert start["state"]["recording"] is True
    assert state["recording"] is True
    assert stop["state"]["recording"] is False
