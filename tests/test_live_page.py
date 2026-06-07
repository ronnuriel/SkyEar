from __future__ import annotations

from pathlib import Path

from server.api import db, get_dashboard_live, get_live_page
from shared.event_schema import AcousticEvent, StationHeartbeat


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
    assert 'href="/static/live.css"' in html


def test_live_js_updates_in_place_without_page_reload():
    text = Path("server/static/live.js").read_text(encoding="utf-8")

    assert "setInterval(refreshLive, REFRESH_MS)" in text
    assert "window.location.reload" not in text
    assert "location.reload" not in text


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

    assert set(payload) == {"server_time", "fusion", "map_state", "stations_health_summary"}
    assert "alerts" not in payload
    assert "stations_latest" not in payload
    assert "events_used" not in payload["fusion"]
    assert payload["map_state"]["bearing_cues"] == []


def test_dashboard_live_can_include_bearing_cues_when_enabled():
    payload = get_dashboard_live(bearing_cues=True)

    assert "bearing_cues" in payload["map_state"]
