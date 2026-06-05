from __future__ import annotations

import numpy as np

from station.two_mic_direction import TwoMicDirectionResult
from station.two_mic_direction import estimate_two_mic_side
from station.two_mic_direction_tracker import TwoMicDirectionTracker, TwoMicDirectionTrackerConfig


def _delayed_pair(delay_samples: int, sample_rate: int = 48000) -> np.ndarray:
    rng = np.random.default_rng(42)
    source = rng.normal(0.0, 0.2, sample_rate // 2).astype(np.float32)
    left = np.zeros_like(source)
    right = np.zeros_like(source)
    if delay_samples >= 0:
        left[:] = source
        right[delay_samples:] = source[: source.size - delay_samples]
    else:
        delay = abs(delay_samples)
        right[:] = source
        left[delay:] = source[: source.size - delay]
    return np.stack([left, right], axis=1)


def test_left_side_when_left_channel_arrives_first():
    result = estimate_two_mic_side(
        _delayed_pair(12),
        48000,
        spacing_m=0.5,
        min_delay_us=40,
        min_peak_ratio=1.1,
    )

    assert result.side == "left"
    assert result.delay_us is not None
    assert result.delay_us > 0
    assert result.confidence and result.confidence > 0.35


def test_right_side_when_right_channel_arrives_first():
    result = estimate_two_mic_side(
        _delayed_pair(-12),
        48000,
        spacing_m=0.5,
        min_delay_us=40,
        min_peak_ratio=1.1,
    )

    assert result.side == "right"
    assert result.delay_us is not None
    assert result.delay_us < 0


def test_center_when_delay_is_small():
    result = estimate_two_mic_side(
        _delayed_pair(0),
        48000,
        spacing_m=0.5,
        min_delay_us=40,
        min_peak_ratio=1.1,
    )

    assert result.side == "center"
    assert abs(float(result.delay_us or 0.0)) < 40.0


def test_two_meter_deadzone_returns_center_for_practical_operator_hint():
    result = estimate_two_mic_side(
        _delayed_pair(48),
        48000,
        spacing_m=2.0,
        center_deadzone_deg=12,
        min_peak_ratio=1.1,
    )

    assert result.side == "center"
    assert result.look_label == "center"
    assert result.angle_from_center_deg is not None
    assert abs(result.angle_from_center_deg) <= 12.0
    assert "LOOK CENTER" in str(result.look_hint)
    assert result.front_back_ambiguous is True


def test_delay_converts_to_signed_angle_and_left_hint():
    result = estimate_two_mic_side(
        _delayed_pair(72),
        48000,
        spacing_m=2.0,
        center_deadzone_deg=12,
        min_peak_ratio=1.1,
    )

    assert result.side == "left"
    assert result.look_label == "left"
    assert result.angle_from_center_deg is not None
    assert 14.0 <= result.angle_from_center_deg <= 16.5
    assert "LOOK LEFT" in str(result.look_hint)
    assert result.sector_width_deg == 60.0


def test_negative_delay_converts_to_right_hint():
    result = estimate_two_mic_side(
        _delayed_pair(-72),
        48000,
        spacing_m=2.0,
        center_deadzone_deg=12,
        min_peak_ratio=1.1,
    )

    assert result.side == "right"
    assert result.look_label == "right"
    assert result.angle_from_center_deg is not None
    assert -16.5 <= result.angle_from_center_deg <= -14.0
    assert "LOOK RIGHT" in str(result.look_hint)


def test_far_side_hint_for_large_angle():
    result = estimate_two_mic_side(
        _delayed_pair(240),
        48000,
        spacing_m=2.0,
        center_deadzone_deg=12,
        far_side_angle_deg=55,
        min_peak_ratio=1.1,
    )

    assert result.side == "left"
    assert result.look_label == "far_left"
    assert "LOOK FAR LEFT" in str(result.look_hint)


def test_front_heading_returns_two_possible_azimuths_not_single_bearing():
    result = estimate_two_mic_side(
        _delayed_pair(72),
        48000,
        spacing_m=2.0,
        center_deadzone_deg=12,
        front_heading_deg=330.0,
        min_peak_ratio=1.1,
    )

    assert result.possible_front_azimuth_deg is not None
    assert result.possible_back_azimuth_deg is not None
    assert result.front_back_ambiguous is True
    assert 344.0 <= result.possible_front_azimuth_deg <= 347.0
    assert 134.0 <= result.possible_back_azimuth_deg <= 137.0


def test_two_mic_tracker_requires_stable_repeated_side():
    tracker = TwoMicDirectionTracker(
        TwoMicDirectionTrackerConfig(
            smoothing_windows=5,
            min_stable_windows=3,
            max_side_flip_rate=0.35,
            min_confidence=0.45,
        )
    )
    first = tracker.update(1.0, _raw_result("left", 15.0))
    second = tracker.update(2.0, _raw_result("left", 16.0))
    third = tracker.update(3.0, _raw_result("left", 14.0))

    assert first.stable is False
    assert second.stable is False
    assert third.stable is True
    assert third.look_label == "left"


def test_two_mic_tracker_suppresses_left_right_flipping():
    tracker = TwoMicDirectionTracker(
        TwoMicDirectionTrackerConfig(
            smoothing_windows=5,
            min_stable_windows=3,
            max_side_flip_rate=0.35,
            min_confidence=0.45,
        )
    )

    tracker.update(1.0, _raw_result("left", 15.0))
    tracker.update(2.0, _raw_result("right", -16.0))
    tracker.update(3.0, _raw_result("left", 14.0))
    frame = tracker.update(4.0, _raw_result("right", -15.0))

    assert frame.stable is False
    assert frame.look_label == "unknown"
    assert "UNSTABLE" in str(frame.look_hint)


def test_uncertain_when_signal_is_too_low():
    result = estimate_two_mic_side(
        np.zeros((4800, 2), dtype=np.float32),
        48000,
        spacing_m=0.5,
    )

    assert result.side == "uncertain"
    assert result.reason == "low_signal"
    assert result.look_label == "unknown"


def _raw_result(side: str, angle: float) -> TwoMicDirectionResult:
    return TwoMicDirectionResult(
        side=side,
        delay_us=angle * 100.0,
        confidence=0.7,
        peak_ratio=2.0,
        angle_from_center_deg=angle,
        look_label=side,
        look_hint=f"LOOK {side.upper()}",
        sector_width_deg=60.0,
        stable=True,
    )
