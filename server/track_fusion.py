from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from server.alert_logic import _same_f0_detected, alert_level_from_recent_events, station_evidence_score
from server.track_observations import observations_from_events
from shared.event_schema import AcousticEvent, FusedAlert, TrackObservation, TrackSummary


DEFAULT_MAX_CANDIDATES_PER_STATION = 3
DEFAULT_FUSION_WINDOW_SEC = 8.0

def _observation_received_time(observation: TrackObservation) -> float:
    return float(observation.server_received_unix or observation.event_timestamp_unix)


def _observation_key(observation: TrackObservation) -> tuple[str, str]:
    candidate_id = observation.candidate_id
    if candidate_id is None:
        candidate_id = observation.source_hint_id
    if candidate_id is None:
        candidate_id = "implicit_event"
    return observation.station_id, str(candidate_id)


def _latest_observations_by_station_candidate(
    observations: list[TrackObservation],
    *,
    max_candidates_per_station: int = DEFAULT_MAX_CANDIDATES_PER_STATION,
) -> list[TrackObservation]:
    latest: dict[tuple[str, str], TrackObservation] = {}
    for observation in observations:
        key = _observation_key(observation)
        existing = latest.get(key)
        if existing is None or _observation_received_time(observation) >= _observation_received_time(existing):
            latest[key] = observation

    by_station: dict[str, list[TrackObservation]] = {}
    for observation in latest.values():
        by_station.setdefault(observation.station_id, []).append(observation)

    selected: list[TrackObservation] = []
    limit = max(1, int(max_candidates_per_station))
    for station_observations in by_station.values():
        station_observations.sort(
            key=lambda item: (
                _observation_evidence_score(item),
                _observation_received_time(item),
            ),
            reverse=True,
        )
        selected.extend(station_observations[:limit])
    return selected


def _observation_metadata(observation: TrackObservation) -> dict[str, Any]:
    return observation.metadata or {}


def _observation_event(observation: TrackObservation) -> AcousticEvent | None:
    return observation.original_event


def _source_id(observation: TrackObservation) -> str | None:
    source_id = observation.source_hint_id
    if source_id is None:
        source_id = _observation_metadata(observation).get("simulated_source_id")
    return None if source_id is None else str(source_id)


def _coverage_radius_m(observation: TrackObservation) -> float | None:
    value = _observation_metadata(observation).get("coverage_radius_m")
    if value is None:
        return None
    return max(0.0, float(value))


def _station_distance_m(left: TrackObservation, right: TrackObservation) -> float | None:
    left_event = _observation_event(left)
    right_event = _observation_event(right)
    if not left_event or not right_event or not left_event.station_location or not right_event.station_location:
        return None
    lat1 = math.radians(float(left_event.station_location.latitude))
    lon1 = math.radians(float(left_event.station_location.longitude))
    lat2 = math.radians(float(right_event.station_location.latitude))
    lon2 = math.radians(float(right_event.station_location.longitude))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371000.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _bearing_deg(from_observation: TrackObservation, to_observation: TrackObservation) -> float | None:
    from_event = _observation_event(from_observation)
    to_event = _observation_event(to_observation)
    if not from_event or not to_event or not from_event.station_location or not to_event.station_location:
        return None
    lat1 = math.radians(float(from_event.station_location.latitude))
    lat2 = math.radians(float(to_event.station_location.latitude))
    dlon = math.radians(float(to_event.station_location.longitude) - float(from_event.station_location.longitude))
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _observation_bearing(observation: TrackObservation) -> float | None:
    value = observation.bearing_deg
    if value is None:
        value = _observation_metadata(observation).get("true_source_bearing_deg")
    if value is None:
        return None
    return float(value) % 360.0


def _angle_delta_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _bearing_lines_plausibly_intersect(left: TrackObservation, right: TrackObservation) -> bool:
    left_bearing = _observation_bearing(left)
    right_bearing = _observation_bearing(right)
    left_to_right = _bearing_deg(left, right)
    right_to_left = _bearing_deg(right, left)
    if left_bearing is None or right_bearing is None or left_to_right is None or right_to_left is None:
        return False
    return _angle_delta_deg(left_bearing, left_to_right) <= 60.0 and _angle_delta_deg(right_bearing, right_to_left) <= 60.0


def _same_sector_or_group(left: TrackObservation, right: TrackObservation) -> bool:
    left_meta = _observation_metadata(left)
    right_meta = _observation_metadata(right)
    for key in ("sector_id", "station_group", "group_id"):
        left_value = left_meta.get(key)
        right_value = right_meta.get(key)
        if left_value is not None and right_value is not None and str(left_value) == str(right_value):
            return True
    return False


def _same_track(left: TrackObservation, right: TrackObservation) -> bool:
    left_source = _source_id(left)
    right_source = _source_id(right)
    if left_source is not None and right_source is not None:
        return left_source == right_source

    if left.station_id == right.station_id:
        return _same_sector_or_group(left, right)

    left_radius = _coverage_radius_m(left)
    right_radius = _coverage_radius_m(right)
    distance = _station_distance_m(left, right)
    if left_radius is not None and right_radius is not None and distance is not None:
        return distance <= left_radius + right_radius or _bearing_lines_plausibly_intersect(left, right)

    if _bearing_lines_plausibly_intersect(left, right):
        return True

    return _same_sector_or_group(left, right)


def _observation_evidence_score(observation: TrackObservation) -> float:
    event = _observation_event(observation)
    event_score = station_evidence_score(event) if event is not None else 0.0
    drone_score = float(observation.drone_score or 0.0)
    confidence = float(observation.confidence or 0.0)
    harmonic_score = float(observation.harmonic_score or 0.0)
    metadata = _observation_metadata(observation)
    suspect_threshold = float(metadata.get("suspect_threshold") or 16.0)
    alert_threshold = float(metadata.get("alert_threshold") or 22.0)
    if alert_threshold <= suspect_threshold:
        alert_threshold = suspect_threshold + 1.0
    harmonic_norm = max(0.0, min(1.0, (harmonic_score - suspect_threshold) / (alert_threshold - suspect_threshold)))
    candidate_score = 0.45 * confidence + 0.35 * drone_score + 0.20 * harmonic_norm
    if drone_score >= 0.60:
        candidate_score = max(candidate_score, 0.60)
    if confidence >= 0.80 and harmonic_norm >= 0.40:
        candidate_score = max(candidate_score, 0.70)
    return float(max(0.0, min(1.0, max(event_score if event and not event.detections else 0.0, candidate_score))))


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


def _unique_events(observations: list[TrackObservation]) -> list[AcousticEvent]:
    events: list[AcousticEvent] = []
    seen: set[tuple[str, float]] = set()
    for observation in observations:
        event = _observation_event(observation)
        if event is None:
            continue
        key = (event.station_id, float(event.timestamp_unix))
        if key in seen:
            continue
        seen.add(key)
        events.append(event)
    return events


def _station_ids(observations: list[TrackObservation]) -> list[str]:
    return sorted({observation.station_id for observation in observations})


def _track_interpretation(observations: list[TrackObservation]) -> str:
    station_ids = _station_ids(observations)
    if len(station_ids) <= 1:
        return "single-station candidate"
    return "multi-station overlapping candidate"


def _estimated_source(observations: list[TrackObservation]) -> dict[str, Any] | None:
    latitudes = []
    longitudes = []
    for observation in observations:
        metadata = _observation_metadata(observation)
        if metadata.get("true_source_latitude") is not None and metadata.get("true_source_longitude") is not None:
            return {
                "latitude": float(metadata["true_source_latitude"]),
                "longitude": float(metadata["true_source_longitude"]),
                "source": "metadata",
            }
        event = _observation_event(observation)
        if event is None:
            continue
        if event.station_location:
            latitudes.append(float(event.station_location.latitude))
            longitudes.append(float(event.station_location.longitude))
    if not latitudes:
        return None
    return {"latitude": sum(latitudes) / len(latitudes), "longitude": sum(longitudes) / len(longitudes), "source": "station_centroid"}


def _track_ambiguity(observations: list[TrackObservation]) -> tuple[int | None, str | None]:
    if len(observations) <= 1:
        return None, None
    station_ids = _station_ids(observations)
    source_ids = {_source_id(observation) for observation in observations if _source_id(observation) is not None}
    if len(station_ids) == 1 and len(observations) > 1:
        return 2, "possible split acoustic source; possible 1-2 targets"
    if len(source_ids) > 1:
        return len(source_ids), "ambiguous multi-target candidate"
    return 1, None


def cluster_observations_into_tracks(
    observations: list[TrackObservation],
    *,
    window_sec: float = DEFAULT_FUSION_WINDOW_SEC,
    max_candidates_per_station: int = DEFAULT_MAX_CANDIDATES_PER_STATION,
) -> list[TrackSummary]:
    now = time.time()
    recent = [observation for observation in observations if now - _observation_received_time(observation) <= window_sec]
    active_observations = [
        observation
        for observation in _latest_observations_by_station_candidate(
            recent,
            max_candidates_per_station=max_candidates_per_station,
        )
        if _observation_evidence_score(observation) > 0.0
    ]
    active_observations.sort(key=lambda observation: (observation.station_id, str(observation.candidate_id or "")))
    if not active_observations:
        return []

    union_find = _UnionFind.create(len(active_observations))
    for idx, left in enumerate(active_observations):
        for other_idx, right in enumerate(active_observations[idx + 1 :], start=idx + 1):
            if _same_track(left, right):
                union_find.union(idx, other_idx)

    grouped: dict[int, list[TrackObservation]] = {}
    for idx, observation in enumerate(active_observations):
        grouped.setdefault(union_find.find(idx), []).append(observation)

    tracks: list[TrackSummary] = []
    sorted_groups = sorted(
        grouped.values(),
        key=lambda items: [(observation.station_id, str(observation.candidate_id or "")) for observation in items],
    )
    for track_index, track_observations in enumerate(sorted_groups):
        track_events = _unique_events(track_observations)
        alert = alert_level_from_recent_events(track_events, window_sec=window_sec)
        track_id = f"track_{chr(ord('A') + track_index)}"
        target_count_hint, ambiguity = _track_ambiguity(track_observations)
        interpretation = _track_interpretation(track_observations)
        if ambiguity:
            interpretation = f"{interpretation}; {ambiguity}"
        tracks.append(
            TrackSummary(
                track_id=track_id,
                station_ids=_station_ids(track_observations),
                events=track_events,
                observations=track_observations,
                target_count_hint=target_count_hint,
                ambiguity=ambiguity,
                level=alert.level,
                confidence=alert.confidence,
                reason=alert.reason,
                estimated_source=_estimated_source(track_observations),
                interpretation=interpretation,
                same_f0=_same_f0_detected(track_events),
            )
        )
    return tracks


def cluster_events_into_tracks(
    events: list[AcousticEvent],
    window_sec: float = DEFAULT_FUSION_WINDOW_SEC,
    max_candidates_per_station: int = DEFAULT_MAX_CANDIDATES_PER_STATION,
) -> list[TrackSummary]:
    return cluster_observations_into_tracks(
        observations_from_events(events),
        window_sec=window_sec,
        max_candidates_per_station=max_candidates_per_station,
    )


def fuse_tracks(
    events: list[AcousticEvent],
    window_sec: float = DEFAULT_FUSION_WINDOW_SEC,
    max_candidates_per_station: int = DEFAULT_MAX_CANDIDATES_PER_STATION,
) -> FusedAlert:
    tracks = cluster_events_into_tracks(
        events,
        window_sec=window_sec,
        max_candidates_per_station=max_candidates_per_station,
    )
    global_level = max((track.level for track in tracks), default=0)
    confidence = max((track.confidence for track in tracks), default=0.0)
    if not tracks:
        interpretation = "background"
    elif len(tracks) == 1:
        interpretation = tracks[0].interpretation
    else:
        interpretation = "multiple local candidates"

    reason = f"{interpretation}; track_count={len(tracks)}, global_level={global_level}"
    for track in tracks:
        reason += f"; {track.track_id}=LEVEL_{track.level} stations={','.join(track.station_ids)} same_f0={'yes' if track.same_f0 else 'no'}"

    events_used = _unique_events([observation for track in tracks for observation in track.observations])[-10:]
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
