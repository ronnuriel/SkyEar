from __future__ import annotations

from typing import Any

from shared.event_schema import AcousticEvent, TrackObservation


def _event_metadata(event: AcousticEvent) -> dict[str, Any]:
    return dict(event.metadata or {})


def _event_received_time(event: AcousticEvent) -> float | None:
    value = event.server_received_unix or (event.metadata or {}).get("server_received_unix")
    return None if value is None else float(value)


def _source_hint_from_metadata(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("source_hint_id")
    if value is None:
        value = metadata.get("simulated_source_id")
    return None if value is None else str(value)


def _event_bearing(event: AcousticEvent) -> float | None:
    value = event.estimated_azimuth_deg
    if value is None:
        value = event.tracked_bearing_deg
    if value is None:
        value = event.raw_bearing_deg
    if value is None:
        value = (event.metadata or {}).get("true_source_bearing_deg")
    if value is None:
        return None
    return float(value) % 360.0


def _event_drone_score(event: AcousticEvent) -> float | None:
    for value in (
        event.combined_drone_evidence_pct,
        event.ml_drone_pct_smoothed,
        event.ml_drone_pct,
        event.hf_p_drone,
        (event.metadata or {}).get("combined_drone_evidence_pct"),
        (event.metadata or {}).get("ml_drone_pct"),
    ):
        if value is not None:
            return max(0.0, min(1.0, float(value)))
    return None


def _candidate_source_hint(candidate_metadata: dict[str, Any], event_metadata: dict[str, Any], explicit: str | None) -> str | None:
    if explicit is not None:
        return str(explicit)
    return _source_hint_from_metadata(candidate_metadata) or _source_hint_from_metadata(event_metadata)


def observations_from_events(events: list[AcousticEvent]) -> list[TrackObservation]:
    observations: list[TrackObservation] = []
    for event in events:
        event_metadata = _event_metadata(event)
        server_received_unix = _event_received_time(event)
        if event.detections:
            for detection in event.detections:
                candidate_metadata = {**event_metadata, **dict(detection.metadata or {})}
                observations.append(
                    TrackObservation(
                        station_id=event.station_id,
                        station_name=event.station_name,
                        event_timestamp_unix=float(event.timestamp_unix),
                        server_received_unix=server_received_unix,
                        candidate_id=detection.candidate_id,
                        confidence=float(detection.confidence),
                        drone_score=detection.drone_score,
                        bearing_deg=None if detection.bearing_deg is None else float(detection.bearing_deg) % 360.0,
                        bearing_error_deg=detection.bearing_error_deg,
                        f0_hz=detection.f0_hz,
                        harmonic_score=detection.harmonic_score,
                        source_hint_id=_candidate_source_hint(
                            candidate_metadata,
                            event_metadata,
                            detection.source_hint_id,
                        ),
                        original_event=event,
                        metadata=candidate_metadata,
                    )
                )
            continue

        observations.append(
            TrackObservation(
                station_id=event.station_id,
                station_name=event.station_name,
                event_timestamp_unix=float(event.timestamp_unix),
                server_received_unix=server_received_unix,
                candidate_id=event.track_id or event_metadata.get("candidate_id"),
                confidence=float(event.confidence or 0.0),
                drone_score=_event_drone_score(event),
                bearing_deg=_event_bearing(event),
                bearing_error_deg=event.bearing_uncertainty_deg,
                f0_hz=event.best_f0_hz,
                harmonic_score=event.harmonic_score,
                source_hint_id=_source_hint_from_metadata(event_metadata),
                original_event=event,
                metadata=event_metadata,
            )
        )
    return observations
