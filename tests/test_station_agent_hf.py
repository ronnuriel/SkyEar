from __future__ import annotations

import numpy as np

from station import station_agent
from station.hf_detector import HFDetectionResult


def test_hf_error_reporter_logs_each_error_once(capsys):
    reporter = station_agent.HFErrorReporter()

    assert reporter.log_once("RuntimeError: first failure") is True
    assert reporter.log_once("RuntimeError: first failure") is False
    assert reporter.log_once("ValueError: second failure") is True

    output = capsys.readouterr().out
    assert output.count("HF error: RuntimeError: first failure") == 1
    assert output.count("HF error: ValueError: second failure") == 1


def test_hf_smoke_test_prints_exact_error(monkeypatch, capsys):
    class FakeDetector:
        model_loaded = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, audio, sample_rate):
            return HFDetectionResult(error="RuntimeError: smoke failed")

    def fake_audio_blocks(**kwargs):
        yield np.zeros((44100, 1), dtype=np.float32)

    monkeypatch.setattr(station_agent, "HFDetector", FakeDetector)
    monkeypatch.setattr(station_agent, "audio_blocks", fake_audio_blocks)

    code = station_agent.run_hf_smoke_test(
        {
            "audio": {"device_id": None, "sample_rate": 44100, "channels": 1},
            "hf": {"model_id": "example/model"},
        }
    )

    output = capsys.readouterr().out
    assert code == 1
    assert "HF error: RuntimeError: smoke failed" in output


def test_hf_smoke_test_prints_prediction(monkeypatch, capsys):
    class FakeDetector:
        model_loaded = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, audio, sample_rate):
            return HFDetectionResult(p_drone=0.42, label="drone")

    def fake_audio_blocks(**kwargs):
        yield np.zeros((44100, 1), dtype=np.float32)

    monkeypatch.setattr(station_agent, "HFDetector", FakeDetector)
    monkeypatch.setattr(station_agent, "audio_blocks", fake_audio_blocks)

    code = station_agent.run_hf_smoke_test(
        {
            "audio": {"device_id": None, "sample_rate": 44100, "channels": 1},
            "hf": {"model_id": "example/model"},
        }
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "p_drone: 0.42" in output
    assert "label: drone" in output


def test_server_base_url_from_events_url():
    assert station_agent.server_base_url_from_events_url("http://host:8080/events") == "http://host:8080"
    assert station_agent.server_base_url_from_events_url("http://host:8080") == "http://host:8080"


def test_startup_connectivity_check_warns_but_returns_failure(monkeypatch):
    def fake_get(url, timeout):
        raise RuntimeError("server down")

    monkeypatch.setattr(station_agent.requests, "get", fake_get)

    ok, reason = station_agent.startup_connectivity_check({"url": "http://host:8080/events"})

    assert ok is False
    assert "RuntimeError" in reason


def test_startup_connectivity_check_can_be_disabled():
    ok, reason = station_agent.startup_connectivity_check(
        {"url": "http://host:8080/events", "startup_check_enabled": False}
    )

    assert ok is True
    assert "disabled" in reason


def test_mic_array_profile_fills_field_positions():
    cfg = {
        "audio": {"channels": 8},
        "mic_array": {"profile": "field_8ch_r0_35m", "sync_mode": "synchronized"},
        "beamforming": {"enabled": True},
    }

    resolved = station_agent.apply_mic_array_profile_defaults(cfg)

    assert len(resolved["mic_array"]["mic_positions_m"]) == 8
    assert resolved["beamforming"]["low_hz"] == 500
    assert resolved["beamforming"]["high_hz"] == 3000


def test_explicit_mic_positions_win_over_profile():
    explicit_positions = [[1.0, 2.0, 3.0]]
    cfg = {
        "mic_array": {
            "profile": "field_8ch_r0_35m",
            "sync_mode": "synchronized",
            "mic_positions_m": explicit_positions,
        },
        "beamforming": {"low_hz": 100, "high_hz": 7000},
    }

    resolved = station_agent.apply_mic_array_profile_defaults(cfg)

    assert resolved["mic_array"]["mic_positions_m"] == explicit_positions
    assert resolved["beamforming"]["low_hz"] == 100
    assert resolved["beamforming"]["high_hz"] == 7000


def test_unknown_mic_array_profile_fails_cleanly():
    cfg = {"mic_array": {"profile": "missing_profile"}}

    try:
        station_agent.apply_mic_array_profile_defaults(cfg)
    except ValueError as exc:
        assert "Unknown mic_array.profile: missing_profile" in str(exc)
    else:
        raise AssertionError("expected ValueError")
