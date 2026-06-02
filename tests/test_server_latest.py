from __future__ import annotations

from shared.event_schema import AcousticEvent
from server.api import get_latest_by_station, get_station_summary, ingest_event
from server.database import db


def _event(station_id: str, timestamp: float, harmonic_score: float = 0.0) -> AcousticEvent:
    return AcousticEvent(
        station_id=station_id,
        timestamp_unix=timestamp,
        status="background",
        confidence=0.1,
        harmonic_score=harmonic_score,
        calibrated=True,
    )


def setup_function():
    db.events.clear()
    db.alerts.clear()


def test_latest_by_station_returns_all_posted_stations():
    ingest_event(_event("station_1", 1.0))
    ingest_event(_event("station_2", 2.0))

    latest = get_latest_by_station()

    assert set(latest) == {"station_1", "station_2"}
    assert latest["station_1"]["timestamp_unix"] == 1.0
    assert latest["station_2"]["timestamp_unix"] == 2.0


def test_latest_by_station_updates_when_new_event_is_posted():
    ingest_event(_event("station_1", 1.0, harmonic_score=1.0))
    ingest_event(_event("station_1", 3.0, harmonic_score=2.0))

    latest = get_latest_by_station()

    assert latest["station_1"]["timestamp_unix"] == 3.0
    assert latest["station_1"]["harmonic_score"] == 2.0


def test_station_summary_returns_compact_latest_state():
    ingest_event(_event("station_1", 1.0))
    ingest_event(_event("station_2", 2.0))

    summary = get_station_summary()

    assert {item["station_id"] for item in summary} == {"station_1", "station_2"}
    assert all("metadata" not in item for item in summary)
