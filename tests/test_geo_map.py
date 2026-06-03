from __future__ import annotations

import time

from dashboard.map_view import normalize_map_state, stations_missing_location
from server.api import get_map_state, ingest_event
from server.database import db
from server.geo import (
    bearing_sector_polygon,
    destination_point,
    estimate_from_recent_bearings,
    haversine_distance_m,
    intersect_bearings,
)
from shared.event_schema import AcousticEvent, EventStatus, GeoPoint
from tools.simulate_geo_events import build_geo_event


def setup_function():
    db.events.clear()
    db.alerts.clear()
    db.heartbeats.clear()


def _candidate_event(
    station_id: str,
    lat: float | None,
    lon: float | None,
    bearing: float | None,
    now: float = 1000.0,
) -> AcousticEvent:
    location = None if lat is None or lon is None else GeoPoint(latitude=lat, longitude=lon, altitude_m=0.0)
    return AcousticEvent(
        station_id=station_id,
        timestamp_unix=now,
        server_received_unix=now,
        station_location=location,
        station_latitude=lat,
        station_longitude=lon,
        station_altitude_m=0.0,
        status=EventStatus.SUSPECT,
        confidence=0.7,
        harmonic_score=18.0,
        ml_drone_pct=0.95,
        combined_drone_evidence_pct=0.5,
        operator_label="ml_drone_candidate",
        candidate_run=2,
        estimated_azimuth_deg=bearing,
        bearing_uncertainty_deg=20.0,
        beam_confidence_pct=0.7,
        calibrated=True,
        metadata={"server_received_unix": now, "operator_label": "ml_drone_candidate"},
    )


def test_destination_point_approximate_movement():
    start_lat, start_lon = 32.0, 34.0
    point = destination_point(start_lat, start_lon, 0.0, 1000.0)

    assert point["latitude"] > start_lat
    assert haversine_distance_m(start_lat, start_lon, point["latitude"], point["longitude"]) == pytest_approx(1000.0, 2.0)


def test_bearing_sector_polygon_has_valid_shape():
    polygon = bearing_sector_polygon(32.0, 34.0, 60.0, 20.0, 25.0, 500.0, steps=8)

    assert len(polygon) == 16
    assert all("latitude" in point and "longitude" in point for point in polygon)


def test_two_bearing_intersection_returns_near_expected_point():
    target = {"latitude": 32.0, "longitude": 34.0}
    station_a = destination_point(target["latitude"], target["longitude"], 180.0, 500.0)
    station_b = destination_point(target["latitude"], target["longitude"], 90.0, 500.0)

    result = intersect_bearings(
        [
            {"station_id": "a", **station_a, "bearing_deg": 0.0},
            {"station_id": "b", **station_b, "bearing_deg": 270.0},
        ]
    )

    assert result is not None
    assert haversine_distance_m(target["latitude"], target["longitude"], result["latitude"], result["longitude"]) < 25.0


def test_one_station_returns_sector_only_no_point():
    event = _candidate_event("a", 32.0, 34.0, 60.0)

    estimate = estimate_from_recent_bearings([event], max_age_sec=10.0, now=1001.0)

    assert estimate["estimate_type"] == "station_sector"
    assert "latitude" not in estimate
    assert estimate["area_polygon"]


def test_no_station_location_returns_no_geo_estimate():
    event = _candidate_event("a", None, None, 60.0)

    estimate = estimate_from_recent_bearings([event], max_age_sec=10.0, now=1001.0)

    assert estimate["estimate_type"] == "none"


def test_map_state_includes_stations_and_estimates():
    now = time.time()
    target = {"latitude": 32.0, "longitude": 34.0}
    station_a = destination_point(target["latitude"], target["longitude"], 180.0, 500.0)
    station_b = destination_point(target["latitude"], target["longitude"], 90.0, 500.0)
    ingest_event(_candidate_event("a", station_a["latitude"], station_a["longitude"], 0.0, now=now))
    ingest_event(_candidate_event("b", station_b["latitude"], station_b["longitude"], 270.0, now=now))

    state = get_map_state()

    assert len(state["stations"]) == 2
    assert state["bearing_cues"]
    assert state["geo_estimates"]
    assert state["geo_estimates"][0]["estimate_type"] in {"bearing_intersection", "multi_station_area"}


def test_dashboard_map_state_parser_handles_missing_optional_fields():
    state = normalize_map_state({"stations": [{"station_id": "a"}]})

    assert stations_missing_location(state)[0]["station_id"] == "a"
    assert state["bearing_cues"] == []


def test_simulated_geo_events_create_map_estimate():
    now = time.time()
    event_a = build_geo_event(station_id="a", lat=31.9955, lon=34.0, bearing=0.0, timestamp=now)
    event_b = build_geo_event(station_id="b", lat=32.0, lon=34.0053, bearing=270.0, timestamp=now)
    ingest_event(event_a)
    ingest_event(event_b)

    state = get_map_state()

    assert len(state["geo_estimates"]) == 1
    assert state["geo_estimates"][0]["latitude"] is not None


def pytest_approx(value: float, abs_tol: float):
    import pytest

    return pytest.approx(value, abs=abs_tol)
