from __future__ import annotations

import argparse
import math
import time
from typing import Any

import requests

from shared.event_schema import AcousticEvent, EventStatus, GeoPoint, StationHeartbeat
from station.station_agent import heartbeat_url_from_events_url


def interpolate_latlon(
    *,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    progress: float,
) -> tuple[float, float]:
    progress = max(0.0, min(1.0, float(progress)))
    return (
        float(start_lat) + (float(end_lat) - float(start_lat)) * progress,
        float(start_lon) + (float(end_lon) - float(start_lon)) * progress,
    )


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_lon = math.radians(float(lon2) - float(lon1))
    x = math.sin(delta_lon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def build_moving_geo_event(
    *,
    station_id: str,
    station_lat: float,
    station_lon: float,
    target_lat: float,
    target_lon: float,
    step_idx: int,
    timestamp: float | None = None,
) -> AcousticEvent:
    timestamp = time.time() if timestamp is None else float(timestamp)
    bearing = bearing_deg(station_lat, station_lon, target_lat, target_lon)
    metadata: dict[str, Any] = {
        "source": "simulate_moving_geo",
        "step_idx": int(step_idx),
        "target_latitude": float(target_lat),
        "target_longitude": float(target_lon),
        "bearing_deg": float(bearing),
        "operator_label": "local_drone_candidate",
        "candidate_run": 3,
        "ml_positive_run": 3,
        "strong_run": 2,
        "note": "synthetic moving two-station bearing test; passive warning/demo data only",
    }
    return AcousticEvent(
        station_id=station_id,
        station_name=f"Simulated {station_id}",
        timestamp_unix=timestamp,
        station_location=GeoPoint(latitude=float(station_lat), longitude=float(station_lon), altitude_m=0.0),
        station_latitude=float(station_lat),
        station_longitude=float(station_lon),
        station_altitude_m=0.0,
        station_location_label="moving geo simulation",
        status=EventStatus.SUSPECT,
        confidence=0.82,
        harmonic_score=24.0,
        harmonic_score_smoothed=23.0,
        harmonic_evidence_pct=0.85,
        harmonic_evidence_pct_smoothed=0.82,
        best_f0_hz=1200,
        f0_family_stable=True,
        ml_drone_pct=0.86,
        ml_drone_pct_smoothed=0.86,
        combined_drone_evidence_pct=0.84,
        hf_p_drone=0.86,
        hf_negative=False,
        hf_positive=True,
        decision_reason="synthetic moving bearing candidate",
        operator_label="local_drone_candidate",
        candidate_run=3,
        ml_positive_run=3,
        strong_run=2,
        estimated_azimuth_deg=float(bearing),
        beamforming_method="simulated_bearing",
        beam_score=0.82,
        beam_confidence_pct=0.82,
        bearing_stable=True,
        bearing_uncertainty_deg=12.0,
        rms=0.04,
        peak=0.4,
        duration_sec=3.0,
        calibrated=True,
        channel_agreement_count=5,
        channel_count=8,
        detector_version="simulate-moving-geo-v1",
        station_mode="simulation",
        metadata=metadata,
    )


def build_heartbeat(event: AcousticEvent) -> StationHeartbeat:
    return StationHeartbeat(
        station_id=event.station_id,
        station_name=event.station_name,
        timestamp_unix=time.time(),
        status="online",
        station_location=event.station_location,
        sample_rate=48000,
        channels=8,
        calibrated=True,
        detector_version="simulate-moving-geo-v1",
        station_mode="simulation",
        last_event_status=str(event.status.value if hasattr(event.status, "value") else event.status),
        last_harmonic_score=event.harmonic_score,
        last_hf_p_drone=event.hf_p_drone,
        metadata={
            "source": "simulate_moving_geo",
            "latitude": event.station_latitude,
            "longitude": event.station_longitude,
            "location_label": event.station_location_label,
            "status": "online",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post a moving two-station geo/bearing simulation.")
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--station-a-lat", type=float, required=True)
    parser.add_argument("--station-a-lon", type=float, required=True)
    parser.add_argument("--station-b-lat", type=float, required=True)
    parser.add_argument("--station-b-lon", type=float, required=True)
    parser.add_argument("--path-start-lat", type=float, required=True)
    parser.add_argument("--path-start-lon", type=float, required=True)
    parser.add_argument("--path-end-lat", type=float, required=True)
    parser.add_argument("--path-end-lon", type=float, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--interval-sec", type=float, default=0.1)
    parser.set_defaults(heartbeat=True)
    parser.add_argument("--heartbeat", dest="heartbeat", action="store_true")
    parser.add_argument("--no-heartbeat", dest="heartbeat", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heartbeat_url = heartbeat_url_from_events_url(args.server)
    steps = max(1, int(args.steps))
    for idx in range(steps):
        progress = 0.0 if steps == 1 else idx / float(steps - 1)
        target_lat, target_lon = interpolate_latlon(
            start_lat=args.path_start_lat,
            start_lon=args.path_start_lon,
            end_lat=args.path_end_lat,
            end_lon=args.path_end_lon,
            progress=progress,
        )
        events = [
            build_moving_geo_event(
                station_id="sim_geo_a",
                station_lat=args.station_a_lat,
                station_lon=args.station_a_lon,
                target_lat=target_lat,
                target_lon=target_lon,
                step_idx=idx,
            ),
            build_moving_geo_event(
                station_id="sim_geo_b",
                station_lat=args.station_b_lat,
                station_lon=args.station_b_lon,
                target_lat=target_lat,
                target_lon=target_lon,
                step_idx=idx,
            ),
        ]
        for event in events:
            response = requests.post(args.server, json=event.model_dump(mode="json"), timeout=3.0)
            response.raise_for_status()
            if args.heartbeat:
                heartbeat_response = requests.post(
                    heartbeat_url,
                    json=build_heartbeat(event).model_dump(mode="json"),
                    timeout=3.0,
                )
                heartbeat_response.raise_for_status()
            print(
                f"posted step={idx} station={event.station_id} "
                f"target={target_lat:.6f},{target_lon:.6f} bearing={event.estimated_azimuth_deg:.1f}"
            )
        if idx + 1 < steps:
            time.sleep(max(0.0, float(args.interval_sec)))


if __name__ == "__main__":
    main()
