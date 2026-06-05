from __future__ import annotations

import pytest

from station.bearing_tracker import (
    BearingTracker,
    BearingTrackerConfig,
    circular_distance_deg,
)


def _tracker(**overrides) -> BearingTracker:
    values = {
        "bearing_smoothing_alpha": 0.25,
        "bearing_flip_hysteresis_windows": 3,
        "max_bearing_jump_deg": 60.0,
        "coast_hold_sec": 5.0,
    }
    values.update(overrides)
    return BearingTracker(BearingTrackerConfig(**values))


def test_bearing_track_wraparound_is_smooth():
    tracker = _tracker()
    frames = [
        tracker.update(
            timestamp=float(idx),
            raw_bearing_deg=bearing,
            bearing_quality="good",
            bearing_reliable=True,
            beam_confidence_pct=0.8,
            peak_ratio=5.0,
            second_peak_ratio=0.2,
        )
        for idx, bearing in enumerate([350, 355, 0, 5, 10])
    ]

    tracked = [frame.tracked_bearing_deg for frame in frames]
    assert all(value is not None for value in tracked)
    assert max(circular_distance_deg(tracked[idx], tracked[idx - 1]) for idx in range(1, len(tracked))) < 10.0
    assert circular_distance_deg(tracked[-1], 0.0) < 20.0
    assert frames[-1].bearing_track_status == "tracking"


def test_unreliable_opposite_lobe_is_suppressed():
    tracker = _tracker()
    tracker.update(timestamp=0.0, raw_bearing_deg=60.0, bearing_quality="good", bearing_reliable=True, peak_ratio=5.0)
    stable = tracker.update(timestamp=1.0, raw_bearing_deg=62.0, bearing_quality="good", bearing_reliable=True, peak_ratio=5.0)
    flip = tracker.update(
        timestamp=2.0,
        raw_bearing_deg=240.0,
        bearing_quality="unreliable",
        bearing_reliable=False,
        peak_ratio=1.4,
        second_peak_ratio=0.9,
    )
    recovered = tracker.update(timestamp=3.0, raw_bearing_deg=63.0, bearing_quality="good", bearing_reliable=True, peak_ratio=5.0)

    assert stable.tracked_bearing_deg is not None
    assert flip.bearing_flip_suppressed is True
    assert flip.bearing_reject_reason == "ambiguous_lobe_flip"
    assert flip.bearing_track_status == "coasting"
    assert circular_distance_deg(flip.tracked_bearing_deg, stable.tracked_bearing_deg) < 1.0
    assert recovered.bearing_flip_suppressed is False
    assert circular_distance_deg(recovered.tracked_bearing_deg, 63.0) < 5.0


def test_reliable_opposite_lobe_switches_after_hysteresis():
    tracker = _tracker(bearing_smoothing_alpha=0.5, bearing_flip_hysteresis_windows=3)
    tracker.update(timestamp=0.0, raw_bearing_deg=60.0, bearing_quality="good", bearing_reliable=True, peak_ratio=5.0)
    tracker.update(timestamp=1.0, raw_bearing_deg=62.0, bearing_quality="good", bearing_reliable=True, peak_ratio=5.0)

    first = tracker.update(timestamp=2.0, raw_bearing_deg=240.0, bearing_quality="good", bearing_reliable=True, peak_ratio=8.0)
    second = tracker.update(timestamp=3.0, raw_bearing_deg=241.0, bearing_quality="good", bearing_reliable=True, peak_ratio=8.0)
    third = tracker.update(timestamp=4.0, raw_bearing_deg=242.0, bearing_quality="good", bearing_reliable=True, peak_ratio=8.0)

    assert first.bearing_flip_suppressed is True
    assert second.bearing_flip_suppressed is True
    assert third.bearing_flip_suppressed is False
    assert third.bearing_reject_reason == "bearing_lobe_switch_confirmed"
    assert circular_distance_deg(third.tracked_bearing_deg, 242.0) < 70.0


def test_circular_distance_uses_short_path():
    assert circular_distance_deg(359.0, 1.0) == pytest.approx(2.0)
