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
    harmonic_evidence_pct: float | None = None,
    ml_drone_pct: float | None = None,
    combined_drone_evidence_pct: float | None = None,
    operator_label: str | None = None,
) -> AcousticEvent:
    return AcousticEvent(
        station_id=station_id,
        timestamp_unix=time.time(),
        status=status,
        confidence=confidence,
        harmonic_score=harmonic_score,
        harmonic_evidence_pct=harmonic_evidence_pct,
        best_f0_hz=best_f0_hz,
        ml_drone_pct=ml_drone_pct,
        combined_drone_evidence_pct=combined_drone_evidence_pct,
        hf_p_drone=hf_p_drone,
        operator_label=operator_label,
        calibrated=True,
        channel_agreement_count=channel_agreement_count,
        channel_count=1,
        metadata={
            "suspect_threshold": 16.0,
            "alert_threshold": 22.0,
            "harmonic_evidence_pct": harmonic_evidence_pct,
            "ml_drone_pct": ml_drone_pct,
            "combined_drone_evidence_pct": combined_drone_evidence_pct,
            "f0_stable": f0_stable,
            "operator_label": operator_label,
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


def test_two_ml_strong_partial_harmonic_stations_elevate_to_level_2():
    events = [
        _event(
            "station_1",
            status="suspect",
            confidence=0.2,
            harmonic_score=17.2,
            harmonic_evidence_pct=0.2,
            hf_p_drone=0.99,
            ml_drone_pct=0.99,
            best_f0_hz=800,
            operator_label="ml_drone_candidate",
        ),
        _event(
            "station_2",
            status="suspect",
            confidence=0.2,
            harmonic_score=17.2,
            harmonic_evidence_pct=0.2,
            hf_p_drone=0.99,
            ml_drone_pct=0.99,
            best_f0_hz=1200,
            operator_label="ml_drone_candidate",
        ),
    ]

    alert = alert_level_from_recent_events(events)

    assert alert.level == 2


def test_one_station_with_strong_combined_evidence_is_level_1():
    alert = alert_level_from_recent_events(
        [
            _event(
                "station_1",
                status="suspect",
                confidence=0.2,
                harmonic_evidence_pct=0.45,
                ml_drone_pct=1.0,
                hf_p_drone=1.0,
                combined_drone_evidence_pct=0.62,
                operator_label="drone_like",
            )
        ]
    )

    assert alert.level == 1


def test_two_stations_with_combined_partial_evidence_are_level_2():
    events = [
        _event(
            "station_1",
            confidence=0.2,
            harmonic_evidence_pct=0.30,
            ml_drone_pct=0.9,
            hf_p_drone=0.9,
            combined_drone_evidence_pct=0.45,
            best_f0_hz=800,
        ),
        _event(
            "station_2",
            confidence=0.2,
            harmonic_evidence_pct=0.30,
            ml_drone_pct=0.9,
            hf_p_drone=0.9,
            combined_drone_evidence_pct=0.45,
            best_f0_hz=1200,
        ),
    ]

    alert = alert_level_from_recent_events(events)

    assert alert.level == 2


def test_two_high_combined_stations_with_same_f0_are_level_3():
    events = [
        _event(
            "station_1",
            harmonic_evidence_pct=0.65,
            ml_drone_pct=0.95,
            hf_p_drone=0.95,
            combined_drone_evidence_pct=0.77,
            best_f0_hz=1000,
        ),
        _event(
            "station_2",
            harmonic_evidence_pct=0.66,
            ml_drone_pct=0.95,
            hf_p_drone=0.95,
            combined_drone_evidence_pct=0.78,
            best_f0_hz=1060,
        ),
    ]

    alert = alert_level_from_recent_events(events)

    assert alert.level == 3


def test_ml_only_without_harmonic_support_stays_level_1_max():
    events = [
        _event(
            "station_1",
            status="background",
            confidence=0.2,
            harmonic_score=0.0,
            harmonic_evidence_pct=0.0,
            hf_p_drone=0.99,
            ml_drone_pct=0.99,
            best_f0_hz=None,
            operator_label="ml_drone_candidate",
        )
    ]

    alert = alert_level_from_recent_events(events)

    assert alert.level <= 1


def test_three_ml_strong_weak_harmonic_stations_are_level_2_not_level_3():
    events = [
        _event(
            f"station_{idx}",
            status="suspect",
            confidence=0.2,
            harmonic_score=16.6,
            harmonic_evidence_pct=0.1,
            hf_p_drone=0.99,
            ml_drone_pct=0.99,
            best_f0_hz=[700, 980, 1260][idx],
            operator_label="ml_drone_candidate",
        )
        for idx in range(3)
    ]

    alert = alert_level_from_recent_events(events)

    assert alert.level == 2


def test_one_alert_station_elevates_to_level_2_not_level_3():
    alert = alert_level_from_recent_events(
        [_event("station_1", status="alert", confidence=0.95, harmonic_score=24.0, hf_p_drone=0.4)]
    )

    assert alert.level == 2
    assert "local_alerts=1" in alert.reason
