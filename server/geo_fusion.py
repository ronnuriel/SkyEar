from __future__ import annotations

from types import SimpleNamespace
import time
from typing import Any

from server.geo import bearing_sector_polygon, estimate_from_recent_bearings, latest_candidate_bearings
from server.track_fusion import fuse_tracks
from shared.event_schema import TrackObservation, TrackSummary


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
                "raw_bearing_deg": item.get("raw_bearing_deg"),
                "tracked_bearing_deg": item.get("tracked_bearing_deg"),
                "uncertainty_deg": item["uncertainty_deg"],
                "beam_confidence_pct": item.get("beam_confidence_pct"),
                "bearing_quality": item.get("bearing_quality"),
                "bearing_reject_reason": item.get("bearing_reject_reason"),
                "bearing_used_for_geo": item.get("bearing_used_for_geo"),
                "coverage_radius_m": item.get("coverage_radius_m"),
                "sector_polygon": bearing_sector_polygon(
                    item["latitude"],
                    item["longitude"],
                    item["bearing_deg"],
                    item["uncertainty_deg"],
                    25.0,
                    float(item.get("coverage_radius_m") or 350.0),
                ),
            }
        )
    return cues


def _track_source_ids(track: TrackSummary) -> list[str]:
    return sorted(
        {
            str(observation.source_hint_id or (observation.metadata or {}).get("simulated_source_id"))
            for observation in track.observations
            if observation.source_hint_id is not None or (observation.metadata or {}).get("simulated_source_id") is not None
        }
    )


def _observation_as_geo_event(observation: TrackObservation) -> Any | None:
    event = observation.original_event
    if event is None:
        return None
    metadata = {
        **dict(event.metadata or {}),
        **dict(observation.metadata or {}),
        "operator_label": "local_drone_candidate",
        "bearing_deg": observation.bearing_deg,
        "estimated_azimuth_deg": observation.bearing_deg,
        "bearing_uncertainty_deg": observation.bearing_error_deg
        or (observation.metadata or {}).get("bearing_uncertainty_deg")
        or getattr(event, "bearing_uncertainty_deg", None)
        or 25.0,
        "bearing_reliable": True,
        "bearing_used_for_geo": True,
    }
    return SimpleNamespace(
        station_id=observation.station_id,
        station_location=event.station_location,
        station_latitude=event.station_latitude,
        station_longitude=event.station_longitude,
        station_altitude_m=event.station_altitude_m,
        timestamp_unix=observation.event_timestamp_unix,
        server_received_unix=observation.server_received_unix,
        operator_label="local_drone_candidate",
        estimated_azimuth_deg=observation.bearing_deg,
        tracked_bearing_deg=observation.bearing_deg,
        raw_bearing_deg=observation.bearing_deg,
        bearing_uncertainty_deg=metadata["bearing_uncertainty_deg"],
        bearing_reliable=True,
        bearing_quality=metadata.get("bearing_quality") or getattr(event, "bearing_quality", None),
        bearing_used_for_geo=True,
        beam_confidence_pct=observation.confidence,
        metadata=metadata,
    )


def geo_estimates_from_tracks(
    tracks: list[TrackSummary],
    *,
    max_age_sec: float = 10.0,
    now: float | None = None,
) -> list[dict[str, Any]]:
    estimates: list[dict[str, Any]] = []
    for track in tracks:
        geo_events = [
            item
            for item in (_observation_as_geo_event(observation) for observation in track.observations)
            if item is not None and item.estimated_azimuth_deg is not None
        ]
        estimate = estimate_from_recent_bearings(geo_events, max_age_sec=max_age_sec, now=now)
        if estimate.get("estimate_type") == "none":
            continue
        estimates.append(
            {
                **estimate,
                "track_id": track.track_id,
                "source_ids": _track_source_ids(track),
                "level": track.level,
                "track_confidence": track.confidence,
                "observation_count": len(track.observations),
            }
        )
    return estimates


def map_tracks_from_tracks(tracks: list[TrackSummary]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in tracks:
        source = track.estimated_source or {}
        source_ids = _track_source_ids(track)
        eta_values = [
            float((observation.metadata or {})["target_eta_sec"])
            for observation in track.observations
            if (observation.metadata or {}).get("target_eta_sec") is not None
        ]
        distance_values = [
            float((observation.metadata or {})["target_distance_to_control_m"])
            for observation in track.observations
            if (observation.metadata or {}).get("target_distance_to_control_m") is not None
        ]
        crossed_lines = [
            str((observation.metadata or {}).get("latest_line_crossed"))
            for observation in track.observations
            if (observation.metadata or {}).get("latest_line_crossed") is not None
        ]
        rows.append(
            {
                "track_id": track.track_id,
                "source_ids": source_ids,
                "station_ids": track.station_ids,
                "level": track.level,
                "confidence": track.confidence,
                "interpretation": track.interpretation,
                "target_count_hint": track.target_count_hint,
                "ambiguity": track.ambiguity,
                "estimated_source": track.estimated_source,
                "latitude": source.get("latitude"),
                "longitude": source.get("longitude"),
                "estimate_source": source.get("source"),
                "observation_count": len(track.observations),
                "latest_line_crossed": crossed_lines[-1] if crossed_lines else None,
                "target_eta_sec": min(eta_values) if eta_values else None,
                "target_distance_to_control_m": min(distance_values) if distance_values else None,
            }
        )
    return rows


def map_tracks_from_events(events: list[Any], max_age_sec: float = 10.0) -> list[dict[str, Any]]:
    return map_tracks_from_tracks(fuse_tracks(events, window_sec=max_age_sec).tracks)


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
            used_for_geo = event.bearing_used_for_geo
            if used_for_geo is None:
                used_for_geo = event_meta.get("bearing_used_for_geo")
            possible_front = event.possible_front_azimuth_deg
            if possible_front is None:
                possible_front = event_meta.get("possible_front_azimuth_deg")
            two_mic_without_heading = bool(event_meta.get("two_mic_direction_enabled")) and possible_front is None
            if used_for_geo is not False and not two_mic_without_heading:
                if event.tracked_bearing_deg is not None:
                    bearing = event.tracked_bearing_deg
                elif event.estimated_azimuth_deg is not None:
                    bearing = event.estimated_azimuth_deg
                elif event_meta.get("tracked_bearing_deg") is not None:
                    bearing = event_meta.get("tracked_bearing_deg")
                else:
                    bearing = event_meta.get("bearing_deg")
        health_label, health_source, last_seen_sec_ago = _map_health(item)
        stations.append(
            {
                "station_id": station_id,
                "name": item.get("station_name") or (event.station_name if event else None),
                "latitude": latitude,
                "longitude": longitude,
                "altitude_m": (event.station_altitude_m if event else None) or event_meta.get("altitude_m") or (heartbeat.get("metadata") or {}).get("altitude_m"),
                "location_label": (event.station_location_label if event else None) or event_meta.get("location_label") or (heartbeat.get("metadata") or {}).get("location_label"),
                "line_id": event_meta.get("line_id") or (heartbeat.get("metadata") or {}).get("line_id"),
                "line_distance_m": event_meta.get("line_distance_m") or (heartbeat.get("metadata") or {}).get("line_distance_m"),
                "fiber_node_id": event_meta.get("fiber_node_id") or (heartbeat.get("metadata") or {}).get("fiber_node_id"),
                "fiber_connected": event_meta.get("fiber_connected") or (heartbeat.get("metadata") or {}).get("fiber_connected"),
                "coverage_radius_m": event_meta.get("coverage_radius_m") or (heartbeat.get("metadata") or {}).get("coverage_radius_m"),
                "last_seen_sec_ago": last_seen_sec_ago,
                "health": health_label,
                "health_source": health_source,
                "last_status": item.get("last_event_status"),
                "operator_label": event.operator_label if event else None,
                "ml_drone_pct": event.ml_drone_pct if event else None,
                "combined_drone_evidence_pct": event.combined_drone_evidence_pct if event else None,
                "candidate_run": event.candidate_run if event else None,
                "bearing_deg": bearing,
                "raw_bearing_deg": event.raw_bearing_deg if event else None,
                "tracked_bearing_deg": event.tracked_bearing_deg if event else None,
                "bearing_track_status": event.bearing_track_status if event else None,
                "bearing_flip_suppressed": event.bearing_flip_suppressed if event else None,
                "bearing_used_for_geo": event.bearing_used_for_geo if event else None,
                "bearing_uncertainty_deg": event.bearing_uncertainty_deg if event else None,
                "beam_confidence_pct": event.beam_confidence_pct if event else None,
                "second_peak_bearing_deg": event.second_peak_bearing_deg if event else None,
                "second_peak_ratio": event.second_peak_ratio if event else None,
                "peak_ratio": event.peak_ratio if event else None,
                "bearing_ambiguity_deg": event.bearing_ambiguity_deg if event else None,
                "bearing_reliable": event.bearing_reliable if event else None,
                "bearing_reject_reason": event.bearing_reject_reason if event else None,
                "bearing_quality": event.bearing_quality if event else None,
            }
        )
    events = db.recent_events(limit=200)
    tracks = fuse_tracks(events, window_sec=fusion_window_sec).tracks
    geo_estimate_suppressed_reason = "multiple_tracks" if len(tracks) > 1 else None
    global_geo_estimates = (
        [] if geo_estimate_suppressed_reason else geo_estimates_from_events(events, max_age_sec=fusion_window_sec, now=now)
    )
    return {
        "server_time": now,
        "stations": stations,
        "bearing_cues": bearing_cues_from_events(events, max_age_sec=fusion_window_sec, now=now),
        "geo_estimates": global_geo_estimates,
        "geo_estimate_suppressed_reason": geo_estimate_suppressed_reason,
        "track_geo_estimates": geo_estimates_from_tracks(tracks, max_age_sec=fusion_window_sec, now=now),
        "tracks": map_tracks_from_tracks(tracks),
    }


def _health_label(alive_state: str | None) -> str:
    if alive_state == "online":
        return "online"
    if alive_state in {"stale", "error"}:
        return "degraded" if alive_state == "error" else "stale"
    return "offline"


def _event_fallback_health(event_age_sec: float | None) -> str:
    if event_age_sec is None:
        return "offline"
    if event_age_sec < 10.0:
        return "online"
    if event_age_sec < 60.0:
        return "stale"
    return "offline"


def _map_health(item: dict[str, Any]) -> tuple[str, str, float | None]:
    heartbeat_age = item.get("heartbeat_age_sec")
    event_age = item.get("event_age_sec")
    if heartbeat_age is not None:
        return _health_label(item.get("alive_state")), "heartbeat", heartbeat_age
    return _event_fallback_health(event_age), "event_fallback", event_age
