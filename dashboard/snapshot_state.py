from __future__ import annotations

import time
from typing import Any, MutableMapping


DEFAULT_REFRESH_MODE = "Manual"
DEBUG_REFRESH_INTERVALS_SEC = {
    "5s": 5.0,
    "10s": 10.0,
    "30s": 30.0,
}


def dashboard_snapshot_key(server_url: str) -> str:
    return f"dashboard_snapshot::{str(server_url).rstrip('/')}"


def should_fetch_dashboard_snapshot(
    session_state: MutableMapping[str, Any],
    server_url: str,
    *,
    refresh_requested: bool = False,
) -> bool:
    if refresh_requested:
        return True
    cached = session_state.get(dashboard_snapshot_key(server_url))
    return not bool(cached and cached.get("payload"))


def store_dashboard_snapshot(
    session_state: MutableMapping[str, Any],
    server_url: str,
    payload: dict[str, Any],
    *,
    loaded_unix: float | None = None,
) -> None:
    session_state[dashboard_snapshot_key(server_url)] = {
        "loaded_unix": time.time() if loaded_unix is None else float(loaded_unix),
        "payload": payload,
    }


def get_dashboard_snapshot(session_state: MutableMapping[str, Any], server_url: str) -> dict[str, Any] | None:
    cached = session_state.get(dashboard_snapshot_key(server_url))
    if not cached:
        return None
    return cached


def snapshot_age_sec(
    session_state: MutableMapping[str, Any],
    server_url: str,
    *,
    now: float | None = None,
) -> float | None:
    cached = get_dashboard_snapshot(session_state, server_url)
    if not cached or cached.get("loaded_unix") is None:
        return None
    return max(0.0, (time.time() if now is None else float(now)) - float(cached["loaded_unix"]))


def is_simulation_station(event: dict[str, Any] | None, health: dict[str, Any] | None = None) -> bool:
    event = event or {}
    health = health or {}
    heartbeat = health.get("heartbeat") or {}
    latest_event = health.get("latest_event") or event
    metadata = {
        **((heartbeat.get("metadata") or {}) if isinstance(heartbeat, dict) else {}),
        **((latest_event.get("metadata") or {}) if isinstance(latest_event, dict) else {}),
        **(event.get("metadata") or {}),
    }
    return (
        str(event.get("station_mode") or latest_event.get("station_mode") or heartbeat.get("station_mode") or "")
        == "simulation"
        or str(metadata.get("source") or "").startswith("simulate_")
        or bool(metadata.get("scenario_id") in {"fiber_grid", "multi_target", "two_near_one_far"})
    )


def should_poll_recording_state(
    event: dict[str, Any] | None,
    health: dict[str, Any] | None,
    *,
    controls_enabled: bool,
) -> bool:
    return bool(controls_enabled) and not is_simulation_station(event, health)


def _recording_summary(health: dict[str, Any] | None) -> str:
    heartbeat = (health or {}).get("heartbeat") or {}
    metadata = heartbeat.get("metadata") or {}
    state = metadata.get("recording_state") or {}
    if not state:
        return "n/a"
    if state.get("recording"):
        duration = float(state.get("duration_sec") or 0.0)
        return f"ON {duration:.0f}s"
    return "OFF"


def compact_station_rows(
    latest_by_station: dict[str, dict[str, Any]],
    health_by_station: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    station_ids = sorted(set(latest_by_station) | set(health_by_station))
    rows = []
    for station_id in station_ids:
        event = latest_by_station.get(station_id) or {}
        health = health_by_station.get(station_id) or {}
        latest_event = health.get("latest_event") or event
        metadata = (event.get("metadata") or {}) or ((latest_event.get("metadata") or {}) if latest_event else {})
        rows.append(
            {
                "station_id": station_id,
                "health": health.get("alive_state") or "offline",
                "last_status": event.get("status") or health.get("last_event_status") or "n/a",
                "last_seen": event.get("server_received_unix")
                or (health.get("heartbeat") or {}).get("server_received_unix")
                or "n/a",
                "confidence": event.get("confidence"),
                "f0": event.get("best_f0_hz") or metadata.get("best_f0_hz"),
                "line_id": metadata.get("line_id"),
                "recording": _recording_summary(health),
                "simulation": is_simulation_station(event, health),
            }
        )
    return rows
