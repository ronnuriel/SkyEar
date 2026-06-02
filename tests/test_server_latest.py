from __future__ import annotations

from shared.event_schema import AcousticEvent, StationHeartbeat
from server.api import get_latest_by_station, get_station_health, get_station_summary, ingest_event, ingest_heartbeat
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
    db.heartbeats.clear()


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


def test_ingest_event_adds_server_received_timestamp_and_latency():
    event = _event("station_1", 1.0)

    response = ingest_event(event)
    latest = get_latest_by_station()["station_1"]

    assert response["ok"] is True
    assert latest["server_received_unix"] is not None
    assert latest["metadata"]["server_received_unix"] == latest["server_received_unix"]
    assert latest["metadata"]["station_to_server_latency_sec"] >= 0.0


def test_station_heartbeat_stores_latest_by_station():
    heartbeat = StationHeartbeat(station_id="station_1", station_name="Station One", timestamp_unix=1.0)

    response = ingest_heartbeat(heartbeat)
    health = get_station_health()["station_1"]

    assert response["ok"] is True
    assert health["station_name"] == "Station One"
    assert health["alive_state"] == "online"
    assert health["heartbeat"]["server_received_unix"] is not None


def test_station_health_reports_online_stale_and_offline_by_heartbeat_age():
    db.add_heartbeat(StationHeartbeat(station_id="online", timestamp_unix=90.0, server_received_unix=95.0))
    db.add_heartbeat(StationHeartbeat(station_id="stale", timestamp_unix=70.0, server_received_unix=80.0))
    db.add_heartbeat(StationHeartbeat(station_id="offline", timestamp_unix=40.0, server_received_unix=60.0))
    db.add_event(_event("missing_heartbeat", 99.0))

    health = db.station_health(now=100.0)

    assert health["online"]["alive_state"] == "online"
    assert health["stale"]["alive_state"] == "stale"
    assert health["offline"]["alive_state"] == "offline"
    assert health["missing_heartbeat"]["alive_state"] == "offline"
