from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from server.alert_logic import _same_f0_detected, alert_level_from_recent_events, station_evidence_score
from shared.event_schema import AcousticEvent, FusedAlert, TrackSummary


def _event_received_time(event: AcousticEvent) -> float:
    return float(event.server_received_unix or (event.metadata or {}).get("server_received_unix") or event.timestamp_unix)


def _latest_by_station(events: list[AcousticEvent]) -> list[AcousticEvent]:
    latest: dict[str, AcousticEvent] = {}
    for event in events:
        existing = latest.get(event.station_id)
        if existing is None or _event_received_time(event) >= _event_received_time(existing):
            latest[event.station_id] = event
    return list(latest.values())


def _metadata(event: AcousticEvent) -> dict[str, Any]:
    return event.metadata or {}


def _source_id(event: AcousticEvent) -> str | None:
    source_id = _metadata(event).get("simulated_source_id")
    return None if source_id is None else str(source_id)


def _scenario_id(event: AcousticEvent) -> str | None:
    scenario_id = _metadata(event).get("scenario_id")
    return None if scenario_id is None else str(scenario_id)


def _coverage_radius_m(event: AcousticEvent) -> float | None:
    value = _metadata(event).get("coverage_radius_m")
    if value is None:
        return None
    return max(0.0, float(value))


def _station_distance_m(left: AcousticEvent, right: AcousticEvent) -> float | None:
    if not left.station_location or not right.station_location:
        return None
    lat1 = math.radians(float(left.station_location.latitude))
    lon1 = math.radians(float(left.station_location.longitude))
    lat2 = math.radians(float(right.station_location.latitude))
    lon2 = math.radians(float(right.station_location.longitude))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371000.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _bearing_deg(from_event: AcousticEvent, to_event: AcousticEvent) -> float | None:
    if not from_event.station_location or not to_event.station_location:
        return None
    lat1 = math.radians(float(from_event.station_location.latitude))
    lat2 = math.radians(float(to_event.station_location.latitude))
    dlon = math.radians(float(to_event.station_location.longitude) - float(from_event.station_location.longitude))
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _event_bearing(event: AcousticEvent) -> float | None:
    value = event.estimated_azimuth_deg
    if value is None:
        value = _metadata(event).get("true_source_bearing_deg")
    if value is None:
        return None
    return float(value) % 360.0


def _angle_delta_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _bearing_lines_plausibly_intersect(left: AcousticEvent, right: AcousticEvent) -> bool:
    left_bearing = _event_bearing(left)
    right_bearing = _event_bearing(right)
    left_to_right = _bearing_deg(left, right)
    right_to_left = _bearing_deg(right, left)
    if left_bearing is None or right_bearing is None or left_to_right is None or right_to_left is None:
        return False
    return _angle_delta_deg(left_bearing, left_to_right) <= 35.0 and _angle_delta_deg(right_bearing, right_to_left) <= 35.0


def _same_sector_or_group(left: AcousticEvent, right: AcousticEvent) -> bool:
    left_meta = _metadata(left)
    right_meta = _metadata(right)
    for key in ("sector_id", "station_group", "group_id"):
        left_value = left_meta.get(key)
        right_value = right_meta.get(key)
        if left_value is not None and right_value is not None and str(left_value) == str(right_value):
            return True
    return False


def _same_track(left: AcousticEvent, right: AcousticEvent) -> bool:
    left_source = _source_id(left)
    right_source = _source_id(right)
    if _scenario_id(left) is not None and left_source is not None and _scenario_id(right) is not None and right_source is not None:
        return left_source == right_source

    left_radius = _coverage_radius_m(left)
    right_radius = _coverage_radius_m(right)
    distance = _station_distance_m(left, right)
    if left_radius is not None and right_radius is not None and distance is not None:
        return distance <= left_radius + right_radius or _bearing_lines_plausibly_intersect(left, right)

    return _same_sector_or_group(left, right)


@dataclass
class _UnionFind:
    parent: list[int]

    @classmethod
    def create(cls, count: int) -> "_UnionFind":
        return cls(parent=list(range(count)))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _track_interpretation(events: list[AcousticEvent]) -> str:
    if len(events) <= 1:
        return "single-station candidate"
    return "multi-station overlapping candidate"


def _estimated_source(events: list[AcousticEvent]) -> dict[str, Any] | None:
    latitudes = []
    longitudes = []
    for event in events:
        metadata = _metadata(event)
        if metadata.get("true_source_latitude") is not None and metadata.get("true_source_longitude") is not None:
            return {
                "latitude": float(metadata["true_source_latitude"]),
                "longitude": float(metadata["true_source_longitude"]),
                "source": "metadata",
            }
        if event.station_location:
            latitudes.append(float(event.station_location.latitude))
            longitudes.append(float(event.station_location.longitude))
    if not latitudes:
        return None
    return {"latitude": sum(latitudes) / len(latitudes), "longitude": sum(longitudes) / len(longitudes), "source": "station_centroid"}


def cluster_events_into_tracks(events: list[AcousticEvent], window_sec: float = 8.0) -> list[TrackSummary]:
    now = time.time()
    recent = [event for event in events if now - _event_received_time(event) <= window_sec]
    active_events = [event for event in _latest_by_station(recent) if station_evidence_score(event) > 0.0]
    active_events.sort(key=lambda event: event.station_id)
    if not active_events:
        return []

    union_find = _UnionFind.create(len(active_events))
    for idx, left in enumerate(active_events):
        for other_idx, right in enumerate(active_events[idx + 1 :], start=idx + 1):
            if _same_track(left, right):
                union_find.union(idx, other_idx)

    grouped: dict[int, list[AcousticEvent]] = {}
    for idx, event in enumerate(active_events):
        grouped.setdefault(union_find.find(idx), []).append(event)

    tracks: list[TrackSummary] = []
    for track_index, track_events in enumerate(sorted(grouped.values(), key=lambda items: [event.station_id for event in items])):
        alert = alert_level_from_recent_events(track_events, window_sec=window_sec)
        track_id = f"track_{chr(ord('A') + track_index)}"
        tracks.append(
            TrackSummary(
                track_id=track_id,
                station_ids=[event.station_id for event in track_events],
                events=track_events,
                level=alert.level,
                confidence=alert.confidence,
                reason=alert.reason,
                estimated_source=_estimated_source(track_events),
                interpretation=_track_interpretation(track_events),
                same_f0=_same_f0_detected(track_events),
            )
        )
    return tracks


def fuse_tracks(events: list[AcousticEvent], window_sec: float = 8.0) -> FusedAlert:
    tracks = cluster_events_into_tracks(events, window_sec=window_sec)
    global_level = max((track.level for track in tracks), default=0)
    confidence = max((track.confidence for track in tracks), default=0.0)
    if not tracks:
        interpretation = "background"
    elif len(tracks) == 1:
        interpretation = "network confirmation candidate" if len(tracks[0].station_ids) > 1 else "single-station candidate"
    else:
        interpretation = "multiple local candidates"

    reason = f"{interpretation}; track_count={len(tracks)}, global_level={global_level}"
    for track in tracks:
        reason += f"; {track.track_id}=LEVEL_{track.level} stations={','.join(track.station_ids)} same_f0={'yes' if track.same_f0 else 'no'}"

    events_used = [event for track in tracks for event in track.events][-10:]
    return FusedAlert(
        timestamp_unix=time.time(),
        level=global_level,
        global_level=global_level,
        status=f"LEVEL_{global_level}",
        confidence=confidence,
        reason=reason,
        interpretation=interpretation,
        tracks=tracks,
        events_used=events_used,
    )
