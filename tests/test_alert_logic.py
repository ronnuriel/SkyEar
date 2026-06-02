from __future__ import annotations

import time

from server.alert_logic import alert_level_from_recent_events
from shared.event_schema import AcousticEvent


def _event(
    station_id: str,
    *,
    status: str = "suspect",
    confidence: float = 0.45,
    harmonic_score: float = 18.0,
    hf_p_drone: float | None = 0.9,
    best_f0_hz: int | None = 1000,
    channel_agreement_count: int = 1,
    f0_stable: bool = False,
) -> AcousticEvent:
    return AcousticEvent(
        station_id=station_id,
        timestamp_unix=time.time(),
        status=status,
        confidence=confidence,
        harmonic_score=harmonic_score,
        best_f0_hz=best_f0_hz,
        hf_p_drone=hf_p_drone,
        calibrated=True,
        channel_agreement_count=channel_agreement_count,
        channel_count=1,
        metadata={
            "suspect_threshold": 16.0,
            "alert_threshold": 22.0,
            "f0_stable": f0_stable,
        },
    )


def test_one_suspect_station_elevates_to_level_1():
    alert = alert_level_from_recent_events([_event("station_1")])

    assert alert.level == 1
    assert "network acoustic confirmation candidate" in alert.reason


def test_two_weak_suspect_stations_with_high_hf_elevate_to_level_2():
    events = [
        _event("station_1", confidence=0.25, harmonic_score=16.5, hf_p_drone=0.95, best_f0_hz=900),
        _event("station_2", confidence=0.25, harmonic_score=16.5, hf_p_drone=0.95, best_f0_hz=1300),
    ]

    alert = alert_level_from_recent_events(events)

    assert alert.level == 2
    assert "active_stations=2" in alert.reason
    assert "max_hf_p_drone=0.95" in alert.reason


def test_three_suspect_stations_with_similar_f0_elevate_to_level_3_candidate():
    events = [
        _event("station_1", best_f0_hz=1000),
        _event("station_2", best_f0_hz=1050),
        _event("station_3", best_f0_hz=1100),
    ]

    alert = alert_level_from_recent_events(events)

    assert alert.level == 3
    assert "same_f0=yes" in alert.reason


def test_background_events_remain_level_0():
    events = [
        _event("station_1", status="background", confidence=0.1, harmonic_score=4.0, hf_p_drone=0.2),
        _event("station_2", status="calibrating", confidence=0.1, harmonic_score=4.0, hf_p_drone=0.2),
    ]

    alert = alert_level_from_recent_events(events)

    assert alert.level == 0
    assert alert.events_used == []


def test_hf_negative_harmonic_sources_do_not_confirm_network():
    events = [
        _event("station_1", confidence=0.9, harmonic_score=27.0, hf_p_drone=0.001, best_f0_hz=800),
        _event("station_2", confidence=0.9, harmonic_score=27.0, hf_p_drone=0.001, best_f0_hz=1300),
    ]

    alert = alert_level_from_recent_events(events)

    assert alert.level < 2
    assert "hf_negative_count=2" in alert.reason


def test_one_alert_station_elevates_to_level_2_not_level_3():
    alert = alert_level_from_recent_events(
        [_event("station_1", status="alert", confidence=0.95, harmonic_score=24.0, hf_p_drone=0.4)]
    )

    assert alert.level == 2
    assert "local_alerts=1" in alert.reason
