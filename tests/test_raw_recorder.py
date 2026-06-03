from __future__ import annotations

import json

import numpy as np

from station.raw_recorder import RawRingBufferRecorder


def test_raw_ring_buffer_saves_wav_and_sidecar(tmp_path):
    recorder = RawRingBufferRecorder(
        directory=tmp_path,
        sample_rate=8000,
        channels=2,
        buffer_seconds=1.0,
        cooldown_seconds=0.0,
    )
    recorder.append(np.zeros((4000, 2), dtype=np.float32))

    saved = recorder.save_candidate("station_1", {"operator_label": "ml_drone_candidate"}, now=100.0)

    assert saved is not None
    wav_path, json_path = saved
    assert wav_path.exists()
    assert json_path.exists()
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["station_id"] == "station_1"
    assert sidecar["operator_label"] == "ml_drone_candidate"
    assert sidecar["channels"] == 2
