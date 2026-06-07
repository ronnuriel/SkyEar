from __future__ import annotations

from pathlib import Path

from dashboard.snapshot_state import (
    DEFAULT_REFRESH_MODE,
    compact_station_rows,
    dashboard_snapshot_key,
    is_simulation_station,
    should_fetch_dashboard_snapshot,
    should_poll_recording_state,
    snapshot_age_sec,
    store_dashboard_snapshot,
)


def test_dashboard_snapshot_helper_uses_cached_state_until_refresh():
    session_state = {}
    server_url = "http://127.0.0.1:8080"

    assert should_fetch_dashboard_snapshot(session_state, server_url) is True

    store_dashboard_snapshot(session_state, server_url, {"server_time": 10.0}, loaded_unix=100.0)

    assert should_fetch_dashboard_snapshot(session_state, server_url) is False
    assert should_fetch_dashboard_snapshot(session_state, server_url, refresh_requested=True) is True
    assert session_state[dashboard_snapshot_key(server_url)]["payload"]["server_time"] == 10.0
    assert snapshot_age_sec(session_state, server_url, now=103.5) == 3.5


def test_dashboard_default_refresh_mode_is_manual():
    assert DEFAULT_REFRESH_MODE == "Manual"


def test_map_title_is_owned_by_map_renderer_only():
    app_text = Path("dashboard/app.py").read_text(encoding="utf-8")
    map_text = Path("dashboard/map_view.py").read_text(encoding="utf-8")

    assert 'st.subheader("Map / Passive Acoustic Situation")' not in app_text
    assert map_text.count('st.subheader("Map / Passive Acoustic Situation")') == 1


def test_recording_state_is_not_polled_for_simulation_stations():
    event = {
        "station_id": "sim_A1",
        "station_mode": "simulation",
        "metadata": {"source": "simulate_fiber_grid", "scenario_id": "fiber_grid"},
    }

    assert is_simulation_station(event, None) is True
    assert should_poll_recording_state(event, None, controls_enabled=True) is False


def test_recording_state_requires_explicit_controls_for_real_station():
    event = {"station_id": "station_001", "metadata": {}}

    assert should_poll_recording_state(event, None, controls_enabled=False) is False
    assert should_poll_recording_state(event, None, controls_enabled=True) is True


def test_compact_station_rows_include_recording_summary_without_polling():
    rows = compact_station_rows(
        {
            "station_001": {
                "station_id": "station_001",
                "status": "background",
                "confidence": 0.42,
                "best_f0_hz": 720.0,
                "server_received_unix": 123.0,
                "metadata": {"line_id": "A"},
            }
        },
        {
            "station_001": {
                "alive_state": "online",
                "heartbeat": {
                    "server_received_unix": 122.0,
                    "metadata": {"recording_state": {"recording": True, "duration_sec": 12.0}},
                },
            }
        },
    )

    assert rows == [
        {
            "station_id": "station_001",
            "health": "online",
            "last_status": "background",
            "last_seen": 123.0,
            "confidence": 0.42,
            "f0": 720.0,
            "line_id": "A",
            "recording": "ON 12s",
            "simulation": False,
        }
    ]
