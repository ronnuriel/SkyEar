from __future__ import annotations

import time
from typing import Any

from server.geo import bearing_sector_polygon, estimate_from_recent_bearings, latest_candidate_bearings


def geo_estimates_from_events(events: list[Any], max_age_sec: float = 10.0, now: float | None = None) -> list[dict[str, Any]]:
    estimate = estimate_from_recent_bearings(events, max_age_sec=max_age_sec, now=now)
    return [] if estimate.get("estimate_type") == "none" else [estimate]


def bearing_cues_from_events(events: list[Any], max_age_sec: float = 10.0, now: float | None = None) -> list[dict[str, Any]]:
    cues = []
    for item in latest_candidate_bearings(events, max_age_sec=max_age_sec, now=now):
        if item.get("bearing_deg") is None:
            continue
        cues.append(
            {
                "station_id": item["station_id"],
                "bearing_deg": item["bearing_deg"],
                "uncertainty_deg": item["uncertainty_deg"],
                "beam_confidence_pct": item.get("beam_confidence_pct"),
                "sector_polygon": bearing_sector_polygon(
                    item["latitude"],
                    item["longitude"],
                    item["bearing_deg"],
                    item["uncertainty_deg"],
                    25.0,
                    800.0,
                ),
            }
        )
    return cues


def map_state_from_db(db, *, now: float | None = None, fusion_window_sec: float = 10.0) -> dict[str, Any]:
    now = time.time() if now is None else float(now)
    health = db.station_health(now=now)
    latest = db.latest_by_station()
    stations = []
    for station_id in sorted(set(health) | set(latest)):
        event = latest.get(station_id)
        item = health.get(station_id, {})
        heartbeat = (item.get("heartbeat") or {}) if item else {}
        event_meta = (event.metadata if event else {}) or {}
        loc = event.station_location if event else None
        latitude = (event.station_latitude if event else None) or event_meta.get("latitude") or (getattr(loc, "latitude", None) if loc else None)
        longitude = (event.station_longitude if event else None) or event_meta.get("longitude") or (getattr(loc, "longitude", None) if loc else None)
        if latitude is None:
            latitude = (heartbeat.get("metadata") or {}).get("latitude")
        if longitude is None:
            longitude = (heartbeat.get("metadata") or {}).get("longitude")
        bearing = None
        if event is not None:
            bearing = event.estimated_azimuth_deg if event.estimated_azimuth_deg is not None else event_meta.get("bearing_deg")
        stations.append(
            {
                "station_id": station_id,
                "name": item.get("station_name") or (event.station_name if event else None),
                "latitude": latitude,
                "longitude": longitude,
                "altitude_m": (event.station_altitude_m if event else None) or event_meta.get("altitude_m") or (heartbeat.get("metadata") or {}).get("altitude_m"),
                "location_label": (event.station_location_label if event else None) or event_meta.get("location_label") or (heartbeat.get("metadata") or {}).get("location_label"),
                "last_seen_sec_ago": item.get("heartbeat_age_sec") if item.get("heartbeat_age_sec") is not None else item.get("event_age_sec"),
                "health": _health_label(item.get("alive_state")),
                "last_status": item.get("last_event_status"),
                "operator_label": event.operator_label if event else None,
                "ml_drone_pct": event.ml_drone_pct if event else None,
                "combined_drone_evidence_pct": event.combined_drone_evidence_pct if event else None,
                "candidate_run": event.candidate_run if event else None,
                "bearing_deg": bearing,
                "bearing_uncertainty_deg": event.bearing_uncertainty_deg if event else None,
                "beam_confidence_pct": event.beam_confidence_pct if event else None,
            }
        )
    events = db.recent_events(limit=200)
    return {
        "server_time": now,
        "stations": stations,
        "bearing_cues": bearing_cues_from_events(events, max_age_sec=fusion_window_sec, now=now),
        "geo_estimates": geo_estimates_from_events(events, max_age_sec=fusion_window_sec, now=now),
        "tracks": [],
    }


def _health_label(alive_state: str | None) -> str:
    if alive_state == "online":
        return "online"
    if alive_state in {"stale", "error"}:
        return "degraded" if alive_state == "error" else "stale"
    return "offline"
