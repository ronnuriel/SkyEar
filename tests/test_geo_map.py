from __future__ import annotations

import sys
import time

from dashboard.map_view import estimate_rows, normalize_map_state, stations_missing_location, track_estimate_rows
from server.api import get_map_state, ingest_event
from server.database import db
from server.geo import (
    bearing_sector_polygon,
    destination_point,
    estimate_from_recent_bearings,
    haversine_distance_m,
    intersect_bearings,
)
from server.geo_fusion import map_state_from_db
from shared.event_schema import AcousticEvent, EventStatus, GeoPoint
from tools.simulate_geo_events import build_geo_event, build_geo_heartbeat, main as simulate_geo_main
from tools.simulate_multi_target import build_events as build_multi_target_events


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
    *,
    raw_bearing: float | None = None,
    tracked_bearing: float | None = None,
    bearing_reliable: bool | None = None,
    bearing_quality: str | None = None,
    bearing_used_for_geo: bool | None = None,
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
        raw_bearing_deg=raw_bearing,
        tracked_bearing_deg=tracked_bearing,
        bearing_uncertainty_deg=20.0,
        bearing_reliable=bearing_reliable,
        bearing_quality=bearing_quality,
        bearing_used_for_geo=bearing_used_for_geo,
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
    assert result["bearing_cross_angle_deg"] == pytest_approx(90.0, 1.0)
    assert result["bearing_geometry_quality"] == "good"


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


def test_unreliable_bearing_is_not_used_for_geo_estimates():
    event = _candidate_event(
        "a",
        32.0,
        34.0,
        None,
        raw_bearing=240.0,
        tracked_bearing=60.0,
        bearing_reliable=False,
        bearing_quality="unreliable",
        bearing_used_for_geo=False,
    )

    estimate = estimate_from_recent_bearings([event], max_age_sec=10.0, now=1001.0)

    assert estimate["estimate_type"] == "none"


def test_two_mic_without_front_heading_is_not_used_for_geo_estimates():
    event = _candidate_event(
        "two_mic",
        32.0,
        34.0,
        60.0,
        raw_bearing=60.0,
        tracked_bearing=60.0,
        bearing_reliable=True,
        bearing_quality="good",
        bearing_used_for_geo=True,
    )
    event.metadata["two_mic_direction_enabled"] = True
    event.metadata["possible_front_azimuth_deg"] = None

    estimate = estimate_from_recent_bearings([event], max_age_sec=10.0, now=1001.0)
    db.add_event(event)
    state = map_state_from_db(db, now=1001.0)

    assert estimate["estimate_type"] == "none"
    assert state["stations"][0]["bearing_deg"] is None


def test_map_state_includes_stations_and_estimates():
    now = time.time()
    target = {"latitude": 32.0, "longitude": 34.0}
    station_a = destination_point(target["latitude"], target["longitude"], 180.0, 500.0)
    station_b = destination_point(target["latitude"], target["longitude"], 90.0, 500.0)
    ingest_event(_candidate_event("a", station_a["latitude"], station_a["longitude"], 0.0, now=now))
    ingest_event(_candidate_event("b", station_b["latitude"], station_b["longitude"], 270.0, now=now))

    state = get_map_state()

    assert len(state["stations"]) == 2
    assert len(state["bearing_cues"]) >= 2
    assert state["geo_estimates"]
    assert state["geo_estimates"][0]["estimate_type"] in {"bearing_intersection", "multi_station_area"}


def test_map_state_exposes_raw_and_tracked_bearings_separately():
    now = time.time()
    db.add_event(
        _candidate_event(
            "tracked",
            32.0,
            34.0,
            82.0,
            now=now,
            raw_bearing=100.0,
            tracked_bearing=82.0,
            bearing_reliable=True,
            bearing_quality="good",
            bearing_used_for_geo=True,
        )
    )

    state = map_state_from_db(db, now=now + 1.0)
    station = state["stations"][0]

    assert station["raw_bearing_deg"] == 100.0
    assert station["tracked_bearing_deg"] == 82.0
    assert station["bearing_deg"] == 82.0


def test_map_state_does_not_expose_precise_bearing_when_unreliable():
    now = time.time()
    db.add_event(
        _candidate_event(
            "unreliable",
            32.0,
            34.0,
            None,
            now=now,
            raw_bearing=240.0,
            tracked_bearing=60.0,
            bearing_reliable=False,
            bearing_quality="unreliable",
            bearing_used_for_geo=False,
        )
    )

    state = map_state_from_db(db, now=now + 1.0)
    station = state["stations"][0]

    assert station["bearing_deg"] is None
    assert station["raw_bearing_deg"] == 240.0
    assert station["tracked_bearing_deg"] == 60.0
    assert state["bearing_cues"] == []
    assert state["geo_estimates"] == []


def test_map_state_uses_recent_event_health_fallback_without_heartbeat():
    now = time.time()
    db.add_event(_candidate_event("event_only", 32.0, 34.0, 60.0, now=now))

    state = map_state_from_db(db, now=now + 8.0)
    station = state["stations"][0]

    assert station["health"] == "online"
    assert station["health_source"] == "event_fallback"


def test_map_state_prefers_heartbeat_health_source():
    now = time.time()
    event = _candidate_event("heartbeat_station", 32.0, 34.0, 60.0, now=now)
    db.add_event(event)
    heartbeat = build_geo_heartbeat(event)
    heartbeat.server_received_unix = now + 1.0
    db.add_heartbeat(heartbeat)

    state = map_state_from_db(db, now=now + 2.0)
    station = state["stations"][0]

    assert station["health"] == "online"
    assert station["health_source"] == "heartbeat"


def test_dashboard_map_state_parser_handles_missing_optional_fields():
    state = normalize_map_state({"stations": [{"station_id": "a"}]})

    assert stations_missing_location(state)[0]["station_id"] == "a"
    assert state["bearing_cues"] == []
    assert state["track_geo_estimates"] == []


def test_map_state_exposes_multi_target_tracks():
    now = time.time()
    for event in build_multi_target_events(timestamp=now):
        db.add_event(event)

    state = map_state_from_db(db, now=now + 1.0)
    station_sets = {tuple(track["station_ids"]) for track in state["tracks"]}

    assert len(state["tracks"]) == 2
    assert ("sim_A1", "sim_A2") in station_sets
    assert ("sim_A2", "sim_A3", "sim_A4") in station_sets
    assert all("track_id" in track for track in state["tracks"])


def test_map_estimate_rows_cap_display_radius_and_keep_raw_radius():
    rows = estimate_rows(
        {
            "geo_estimates": [
                {
                    "latitude": 32.0,
                    "longitude": 34.0,
                    "radius_m": 1200.0,
                    "confidence": 0.2,
                    "bearing_geometry_quality": "poor",
                }
            ]
        }
    )

    assert rows[0]["raw_radius_m"] == 1200.0
    assert rows[0]["display_radius_m"] == 350.0
    assert rows[0]["fill_color"][0] == 140


def test_track_estimate_rows_use_track_labels_and_confidence_color():
    rows = track_estimate_rows(
        {
            "track_geo_estimates": [
                {
                    "track_id": "track_A",
                    "latitude": 32.0,
                    "longitude": 34.0,
                    "radius_m": 260.0,
                    "confidence": 0.72,
                    "level": 2,
                    "bearing_geometry_quality": "good",
                }
            ]
        }
    )

    assert rows[0]["label"] == "track_A"
    assert rows[0]["display_radius_m"] == 260.0
    assert rows[0]["fill_color"][0] == 220


def test_simulated_geo_events_create_map_estimate():
    now = time.time()
    event_a = build_geo_event(station_id="a", lat=31.9955, lon=34.0, bearing=0.0, timestamp=now)
    event_b = build_geo_event(station_id="b", lat=32.0, lon=34.0053, bearing=270.0, timestamp=now)
    ingest_event(event_a)
    ingest_event(event_b)

    state = get_map_state()

    assert len(state["geo_estimates"]) == 1
    assert state["geo_estimates"][0]["latitude"] is not None


def test_simulate_geo_events_cli_posts_events_and_heartbeats(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate_geo_events",
            "--server",
            "http://server:8080/events",
            "--station-a-lat",
            "31.9955",
            "--station-a-lon",
            "34.0",
            "--station-a-bearing",
            "0",
            "--station-b-lat",
            "32.0",
            "--station-b-lon",
            "34.0053",
            "--station-b-bearing",
            "270",
        ],
    )

    simulate_geo_main()

    assert [call[0] for call in calls] == [
        "http://server:8080/events",
        "http://server:8080/stations/heartbeat",
        "http://server:8080/events",
        "http://server:8080/stations/heartbeat",
    ]
    assert calls[1][1]["metadata"]["latitude"] == 31.9955
    assert calls[1][1]["metadata"]["status"] == "online"


def test_simulate_geo_events_cli_can_disable_heartbeats(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

    monkeypatch.setattr("requests.post", lambda url, json, timeout: calls.append((url, json)) or Response())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate_geo_events",
            "--server",
            "http://server:8080/events",
            "--station-a-lat",
            "31.9955",
            "--station-a-lon",
            "34.0",
            "--station-a-bearing",
            "0",
            "--station-b-lat",
            "32.0",
            "--station-b-lon",
            "34.0053",
            "--station-b-bearing",
            "270",
            "--no-heartbeat",
        ],
    )

    simulate_geo_main()

    assert [call[0] for call in calls] == ["http://server:8080/events", "http://server:8080/events"]


def test_poor_bearing_cross_angle_lowers_confidence():
    now = 1000.0
    target = {"latitude": 32.0, "longitude": 34.0}
    good_station_a = destination_point(target["latitude"], target["longitude"], 180.0, 500.0)
    good_station_b = destination_point(target["latitude"], target["longitude"], 90.0, 500.0)
    poor_station_a = destination_point(target["latitude"], target["longitude"], 180.0, 500.0)
    poor_station_b = destination_point(target["latitude"], target["longitude"], 190.0, 500.0)

    good = estimate_from_recent_bearings(
        [
            _candidate_event("good_a", good_station_a["latitude"], good_station_a["longitude"], 0.0, now=now),
            _candidate_event("good_b", good_station_b["latitude"], good_station_b["longitude"], 270.0, now=now),
        ],
        max_age_sec=10.0,
        now=now + 1.0,
    )
    poor = estimate_from_recent_bearings(
        [
            _candidate_event("poor_a", poor_station_a["latitude"], poor_station_a["longitude"], 0.0, now=now),
            _candidate_event("poor_b", poor_station_b["latitude"], poor_station_b["longitude"], 10.0, now=now),
        ],
        max_age_sec=10.0,
        now=now + 1.0,
    )

    assert good["bearing_geometry_quality"] == "good"
    assert poor["bearing_geometry_quality"] == "poor"
    assert poor["bearing_cross_angle_deg"] < 20.0
    assert poor["confidence"] < good["confidence"]


def pytest_approx(value: float, abs_tol: float):
    import pytest

    return pytest.approx(value, abs=abs_tol)
