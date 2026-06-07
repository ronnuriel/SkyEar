from __future__ import annotations

from pathlib import Path

from server.api import db, get_dashboard_live, get_live_config, get_live_page
from server.live_config import live_map_config_from_mapping
from shared.event_schema import AcousticEvent, StationHeartbeat
from tools.simulate_fiber_grid import simulate_fiber_grid


def setup_function():
    db.events.clear()
    db.alerts.clear()
    db.heartbeats.clear()


def test_live_page_serves_html_and_static_assets():
    response = get_live_page()
    html_path = Path(response.path)
    js_path = Path("server/static/live.js")
    css_path = Path("server/static/live.css")

    assert html_path.name == "live.html"
    assert html_path.exists()
    assert js_path.exists()
    assert css_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert 'src="/static/live.js"' in html
    assert 'src="/static/live_map.js"' in html
    assert 'href="/static/live.css"' in html
    assert 'id="togglePause"' in html
    assert 'id="lineLayer"' in html
    assert 'id="nearestEta"' in html
    assert 'id="mapMode"' in html
    assert 'id="tileLayer"' in html


def test_live_js_updates_in_place_without_page_reload():
    text = Path("server/static/live.js").read_text(encoding="utf-8")

    assert "setInterval(refreshLive, REFRESH_MS)" in text
    assert "window.location.reload" not in text
    assert "location.reload" not in text
    assert "updateStationLines" in text
    assert "togglePause" in text


def test_live_map_js_supports_geo_query_params_and_renderers():
    text = Path("server/static/live_map.js").read_text(encoding="utf-8")

    assert "queryOptions" in text
    assert 'params.get("mode")' in text
    assert 'params.get("lat")' in text
    assert 'params.get("lon")' in text
    assert 'params.get("zoom")' in text
    assert "class SchematicRenderer" in text
    assert "class GeoRenderer" in text


def test_dashboard_live_endpoint_is_compact_http_payload():
    now = 1000.0
    db.add_event(
        AcousticEvent(
            station_id="live_station",
            timestamp_unix=now,
            server_received_unix=now,
            status="suspect",
            confidence=0.7,
            harmonic_score=18.0,
            calibrated=True,
            metadata={"server_received_unix": now},
        )
    )
    db.add_heartbeat(StationHeartbeat(station_id="live_station", timestamp_unix=now, server_received_unix=now))
    payload = get_dashboard_live()

    assert {
        "server_time",
        "fusion",
        "map_state",
        "stations_health_summary",
        "tracks",
        "track_count",
        "nearest_eta_sec",
        "latest_line_crossed",
        "online_station_count",
        "total_station_count",
        "degraded_station_count",
        "offline_station_count",
        "source_ids",
    } <= set(payload)
    assert "alerts" not in payload
    assert "stations_latest" not in payload
    assert payload["fusion"]["track_count"] == payload["track_count"]
    assert "events_used" not in payload["fusion"]
    assert "observations" not in (payload["tracks"][0] if payload["tracks"] else {})
    assert payload["map_state"]["bearing_cues"] == []


def test_dashboard_live_can_include_bearing_cues_when_enabled():
    payload = get_dashboard_live(bearing_cues=True)

    assert "bearing_cues" in payload["map_state"]


def test_live_config_defaults_to_schematic_without_tiles():
    payload = get_live_config()

    assert payload["live_map"]["mode"] == "schematic"
    assert payload["live_map"]["tile_url"] is None


def test_live_map_config_ignores_external_tiles_unless_enabled():
    config = live_map_config_from_mapping(
        {"live_map": {"mode": "geo", "tile_url": "https://tiles.example/{z}/{x}/{y}.png"}}
    )

    assert config["mode"] == "schematic"
    assert config["tile_url"] is None
    assert config["warning"]


def test_live_map_config_can_enable_online_osm_tiles():
    config = live_map_config_from_mapping({"live_map": {"mode": "geo", "allow_online_tiles": True}})

    assert config["mode"] == "geo"
    assert "tile.openstreetmap.org" in config["tile_url"]


def test_dashboard_live_includes_fiber_grid_tactical_fields():
    simulation = simulate_fiber_grid(steps=40, targets=2, target_separation_m=300.0)
    for event in simulation.events:
        db.add_event(event)
    for heartbeat in simulation.heartbeats[-19:]:
        db.add_heartbeat(heartbeat)

    payload = get_dashboard_live(bearing_cues=True)

    assert len(payload["tracks"]) == 2
    assert payload["track_count"] == 2
    assert payload["nearest_eta_sec"] is not None
    assert payload["latest_line_crossed"] in {"A", "B", "C"}
    assert payload["online_station_count"] + payload["degraded_station_count"] + payload["offline_station_count"] == payload["total_station_count"]
    assert {tuple(track["source_ids"]) for track in payload["tracks"]} == {("T1",), ("T2",)}
    assert all("observations" not in track for track in payload["tracks"])
    assert payload["map_state"]["tracks"] == payload["tracks"]
    assert payload["map_state"]["bearing_cues"]
    assert "alerts" not in payload
    assert "recording_state" not in str(payload)


def test_fiber_grid_custom_control_point_reaches_live_payload():
    simulation = simulate_fiber_grid(steps=1, targets=1, control_lat=31.9, control_lon=35.1)
    for event in simulation.events:
        db.add_event(event)
    for heartbeat in simulation.heartbeats[-19:]:
        db.add_heartbeat(heartbeat)

    payload = get_dashboard_live()
    control = payload["map_state"]["control_point"]

    assert control["latitude"] == 31.9
    assert control["longitude"] == 35.1
    assert all(station["latitude"] is not None and station["longitude"] is not None for station in payload["map_state"]["stations"])
    assert {station["line_id"] for station in payload["map_state"]["stations"]} == {"A", "B", "C"}
