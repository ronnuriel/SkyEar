from __future__ import annotations

import numpy as np

from station.detector_state import StationDetectorState, StationDetectorStateConfig


SR = 44100


def _cfg(**overrides) -> StationDetectorStateConfig:
    values = {
        "calibration_seconds": 2.0,
        "min_alert_duration_sec": 2.0,
        "clear_after_sec": 1.0,
        "stability_history_windows": 4,
        "stability_min_score_windows": 3,
        "stability_max_f0_std_hz": 80.0,
    }
    values.update(overrides)
    return StationDetectorStateConfig(**values)


def _quiet() -> np.ndarray:
    return np.zeros(SR, dtype=np.float32)


def _harmonic(f0: float) -> np.ndarray:
    t = np.arange(SR, dtype=np.float32) / SR
    audio = np.zeros_like(t)
    for k in range(1, 5):
        audio += (0.05 / k) * np.sin(2 * np.pi * f0 * k * t)
    return audio.astype(np.float32)


def _calibrated_state(**overrides) -> StationDetectorState:
    state = StationDetectorState(_cfg(**overrides))
    state.update(_quiet(), SR, 0.0)
    state.update(_quiet(), SR, 2.0)
    return state


def test_stable_harmonic_f0_allows_alert_after_duration():
    state = _calibrated_state()

    state.update(_harmonic(900), SR, 3.0)
    state.update(_harmonic(900), SR, 4.0)
    frame = state.update(_harmonic(900), SR, 5.2)

    assert frame.f0_stable is True
    assert frame.status == "alert"


def test_changing_f0_stays_below_alert_quickly():
    state = _calibrated_state()

    state.update(_harmonic(700), SR, 3.0)
    state.update(_harmonic(1000), SR, 4.0)
    frame = state.update(_harmonic(1300), SR, 5.2)

    assert frame.f0_stable is False
    assert frame.status in {"suspect", "drone_like"}


def test_hf_only_still_does_not_alert_with_stability_enabled():
    state = _calibrated_state()

    frame = state.update(_quiet(), SR, 3.0, hf_p_drone=0.99)

    assert frame.status == "suspect"
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.harmonic_score < frame.suspect_threshold
