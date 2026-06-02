from __future__ import annotations

from station.detector_state import StationDetectorState, StationDetectorStateConfig
from tools.simulate_station import generate_synthetic_audio


SR = 16000
WINDOW_SEC = 0.5


def _state() -> StationDetectorState:
    return StationDetectorState(
        StationDetectorStateConfig(
            f0_min=500,
            f0_max=1500,
            max_freq=5000,
            calibration_seconds=1.0,
            min_alert_duration_sec=1.0,
            clear_after_sec=1.0,
        )
    )


def _calibrate(state: StationDetectorState) -> None:
    state.update(generate_synthetic_audio("background", 0.0, SR, WINDOW_SEC, 8), SR, 0.0)
    state.update(generate_synthetic_audio("background", 1.0, SR, WINDOW_SEC, 8), SR, 1.0)


def test_synthetic_drone_hover_produces_suspect_or_alert_after_calibration():
    state = _state()
    _calibrate(state)

    state.update(generate_synthetic_audio("drone_hover", 10.0, SR, WINDOW_SEC, 8), SR, 2.0)
    frame = state.update(generate_synthetic_audio("drone_hover", 11.0, SR, WINDOW_SEC, 8), SR, 3.1)

    assert frame.status in {"suspect", "drone_like", "alert"}
    assert frame.agreement_count > 0


def test_synthetic_background_remains_background_after_calibration():
    state = _state()
    _calibrate(state)

    frame = state.update(generate_synthetic_audio("background", 10.0, SR, WINDOW_SEC, 8), SR, 2.0)

    assert frame.status == "background"


def test_synthetic_multi_rotor_jitter_returns_eight_channel_evidence():
    state = _state()
    _calibrate(state)

    frame = state.update(generate_synthetic_audio("multi_rotor_jitter", 10.0, SR, WINDOW_SEC, 8), SR, 2.0)

    assert frame.channel_count == 8
    assert len(frame.per_channel) == 8
    assert frame.agreement_count > 0
