from __future__ import annotations

import argparse
import time

import requests

from shared.event_schema import AcousticEvent, EventStatus, GeoPoint


def build_geo_event(
    *,
    station_id: str,
    lat: float,
    lon: float,
    bearing: float,
    timestamp: float | None = None,
) -> AcousticEvent:
    timestamp = time.time() if timestamp is None else float(timestamp)
    return AcousticEvent(
        station_id=station_id,
        station_name=station_id,
        timestamp_unix=timestamp,
        station_location=GeoPoint(latitude=float(lat), longitude=float(lon), altitude_m=0.0),
        station_latitude=float(lat),
        station_longitude=float(lon),
        station_altitude_m=0.0,
        station_heading_offset_deg=0.0,
        station_location_label="simulated",
        status=EventStatus.SUSPECT,
        confidence=0.72,
        harmonic_score=18.0,
        harmonic_evidence_pct=0.35,
        harmonic_evidence_pct_smoothed=0.35,
        ml_drone_pct=0.95,
        combined_drone_evidence_pct=0.52,
        hf_p_drone=0.95,
        operator_label="ml_drone_candidate",
        candidate_run=2,
        estimated_azimuth_deg=float(bearing) % 360.0,
        beam_confidence_pct=0.75,
        bearing_stable=True,
        bearing_uncertainty_deg=18.0,
        rms=0.02,
        calibrated=True,
        detector_version="simulated-geo-events-v1",
        metadata={
            "source": "simulate_geo_events",
            "bearing_deg": float(bearing) % 360.0,
            "bearing_uncertainty_deg": 18.0,
            "beam_confidence_pct": 0.75,
            "operator_label": "ml_drone_candidate",
            "note": "synthetic passive bearing cue for map testing; not targeting-grade",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post synthetic passive geo events for SkyEar map testing.")
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--station-a-lat", type=float, required=True)
    parser.add_argument("--station-a-lon", type=float, required=True)
    parser.add_argument("--station-a-bearing", type=float, required=True)
    parser.add_argument("--station-b-lat", type=float, required=True)
    parser.add_argument("--station-b-lon", type=float, required=True)
    parser.add_argument("--station-b-bearing", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = [
        build_geo_event(
            station_id="geo_station_a",
            lat=args.station_a_lat,
            lon=args.station_a_lon,
            bearing=args.station_a_bearing,
        ),
        build_geo_event(
            station_id="geo_station_b",
            lat=args.station_b_lat,
            lon=args.station_b_lon,
            bearing=args.station_b_bearing,
        ),
    ]
    for event in events:
        response = requests.post(args.server, json=event.model_dump(mode="json"), timeout=2.0)
        response.raise_for_status()
        print(f"posted {event.station_id} bearing={event.estimated_azimuth_deg}")


if __name__ == "__main__":
    main()
