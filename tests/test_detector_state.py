from __future__ import annotations

import numpy as np

from station.detector_state import StationDetectorState, StationDetectorStateConfig


SR = 44100


def _cfg(**overrides) -> StationDetectorStateConfig:
    values = {
        "calibration_seconds": 2.0,
        "min_alert_duration_sec": 3.0,
        "clear_after_sec": 2.5,
        "min_suspect_threshold": 14.0,
        "min_alert_threshold": 22.0,
    }
    values.update(overrides)
    return StationDetectorStateConfig(**values)


def _quiet(sec: float = 1.0, channels: int = 1) -> np.ndarray:
    audio = np.zeros(int(SR * sec), dtype=np.float32)
    if channels == 1:
        return audio
    return np.repeat(audio.reshape(-1, 1), channels, axis=1)


def _harmonic(sec: float = 1.0, channels: int = 1) -> np.ndarray:
    t = np.arange(int(SR * sec), dtype=np.float32) / SR
    mono = np.zeros_like(t)
    for k in range(1, 5):
        mono += (0.05 / k) * np.sin(2 * np.pi * 700 * k * t)
    if channels == 1:
        return mono.astype(np.float32)
    out = np.zeros((mono.size, channels), dtype=np.float32)
    out[:, 0] = mono
    out[:, 1:] = mono[:, None] * 0.8
    return out


def _calibrated_state(**overrides) -> StationDetectorState:
    state = StationDetectorState(_cfg(**overrides))
    state.update(_quiet(), SR, 0.0)
    state.update(_quiet(), SR, 2.0)
    return state


def test_during_calibration_status_is_calibrating():
    state = StationDetectorState(_cfg())

    frame = state.update(_quiet(), SR, 0.0)

    assert frame.status == "calibrating"
    assert frame.calibrated is False


def test_after_calibration_quiet_audio_becomes_background():
    state = StationDetectorState(_cfg())

    state.update(_quiet(), SR, 0.0)
    frame = state.update(_quiet(), SR, 2.0)

    assert frame.status == "background"
    assert frame.calibrated is True


def test_harmonic_stack_triggers_alert_only_after_min_duration():
    state = _calibrated_state()

    first = state.update(_harmonic(), SR, 3.0)
    second = state.update(_harmonic(), SR, 5.0)
    third = state.update(_harmonic(), SR, 6.1)

    assert first.status in {"suspect", "drone_like"}
    assert second.status in {"suspect", "drone_like"}
    assert third.status == "alert"


def test_signal_clears_to_background_after_clean_windows():
    state = _calibrated_state(clear_after_sec=2.5)

    state.update(_harmonic(), SR, 3.0)
    state.update(_harmonic(), SR, 4.5)
    state.update(_harmonic(), SR, 6.1)
    holding = state.update(_quiet(), SR, 7.0)
    cleared = state.update(_quiet(), SR, 9.0)

    assert holding.status == "alert"
    assert cleared.status == "background"


def test_multichannel_input_returns_channel_scores_and_agreement_count():
    state = _calibrated_state()

    frame = state.update(_harmonic(channels=2), SR, 3.0)

    assert frame.channel_count == 2
    assert len(frame.per_channel) == 2
    assert frame.strongest_channel in {0, 1}
    assert frame.agreement_count == 2


def test_hf_support_alone_does_not_trigger_alert():
    state = _calibrated_state()

    frame = state.update(_quiet(), SR, 3.0, hf_p_drone=0.99, cnn_p_drone=0.99)

    assert frame.status == "background"
    assert frame.harmonic_score < frame.suspect_threshold


def test_hf_negative_caps_single_channel_alert_harmonic_to_suspect():
    state = _calibrated_state()

    frame = state.update(_harmonic(), SR, 3.0, hf_p_drone=0.001)

    assert frame.harmonic_score >= frame.alert_threshold
    assert frame.hf_negative is True
    assert frame.hf_positive is False
    assert frame.status == "suspect"


def test_hf_negative_caps_single_channel_even_when_f0_stable():
    state = _calibrated_state()

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.001)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.001)
    frame = state.update(_harmonic(), SR, 6.1, hf_p_drone=0.001)

    assert frame.harmonic_score >= frame.alert_threshold
    assert frame.f0_stable is True
    assert frame.hf_negative is True
    assert frame.status == "suspect"


def test_hf_positive_with_high_harmonic_can_reach_alert_after_duration():
    state = _calibrated_state()

    first = state.update(_harmonic(), SR, 3.0, hf_p_drone=0.95)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.95)
    third = state.update(_harmonic(), SR, 6.1, hf_p_drone=0.95)

    assert first.status in {"suspect", "drone_like"}
    assert third.hf_positive is True
    assert third.f0_stable is True
    assert third.status == "alert"


def test_hf_missing_single_channel_keeps_existing_harmonic_behavior():
    state = _calibrated_state()

    state.update(_harmonic(), SR, 3.0)
    state.update(_harmonic(), SR, 4.5)
    frame = state.update(_harmonic(), SR, 6.1)

    assert frame.harmonic_score >= frame.alert_threshold
    assert frame.f0_stable is True
    assert frame.hf_negative is False
    assert frame.hf_positive is False
    assert frame.status in {"suspect", "drone_like", "alert"}


def test_multichannel_agreement_and_stable_f0_can_alert_without_hf():
    state = _calibrated_state()

    state.update(_harmonic(channels=2), SR, 3.0)
    state.update(_harmonic(channels=2), SR, 4.5)
    frame = state.update(_harmonic(channels=2), SR, 6.1)

    assert frame.agreement_count >= 2
    assert frame.f0_stable is True
    assert frame.status == "alert"


def test_multichannel_agreement_and_stable_f0_can_alert_despite_negative_hf():
    state = _calibrated_state()

    state.update(_harmonic(channels=2), SR, 3.0, hf_p_drone=0.001)
    state.update(_harmonic(channels=2), SR, 4.5, hf_p_drone=0.001)
    frame = state.update(_harmonic(channels=2), SR, 6.1, hf_p_drone=0.001)

    assert frame.hf_negative is True
    assert frame.agreement_count >= 2
    assert frame.f0_stable is True
    assert frame.status == "alert"
