from __future__ import annotations

import argparse
import time
from typing import Any

import requests

from shared.event_schema import AcousticEvent, EventStatus, GeoPoint, StationHeartbeat
from station.station_agent import heartbeat_url_from_events_url


SCENARIO_ID = "two_near_one_far"
NEAR_SOURCE_ID = "shared_near_source"
FAR_SOURCE_ID = "independent_far_source"


def build_simulated_event(
    station_id: str,
    *,
    latitude: float,
    longitude: float,
    source_id: str,
    coverage_radius_m: float,
    bearing_deg: float,
    f0_hz: int = 1040,
    timestamp: float | None = None,
) -> AcousticEvent:
    timestamp = time.time() if timestamp is None else float(timestamp)
    metadata: dict[str, Any] = {
        "source": "simulate_two_near_one_far",
        "scenario_id": SCENARIO_ID,
        "simulated_source_id": source_id,
        "coverage_radius_m": float(coverage_radius_m),
        "suspect_threshold": 16.0,
        "alert_threshold": 22.0,
        "harmonic_evidence_pct": 0.92,
        "harmonic_evidence_pct_smoothed": 0.90,
        "ml_drone_pct": 0.96,
        "combined_drone_evidence_pct": 0.93,
        "f0_stable": True,
        "f0_family_stable": True,
        "candidate_run": 4,
        "ml_positive_run": 4,
        "strong_run": 3,
        "note": "synthetic spatial fusion test; passive warning/demo data only",
    }
    return AcousticEvent(
        station_id=station_id,
        station_name=f"Simulated {station_id}",
        timestamp_unix=timestamp,
        station_location=GeoPoint(latitude=latitude, longitude=longitude, altitude_m=0.0),
        station_latitude=latitude,
        station_longitude=longitude,
        station_altitude_m=0.0,
        station_location_label="two-near-one-far simulation",
        status=EventStatus.ALERT,
        confidence=0.92,
        harmonic_score=28.0,
        harmonic_score_smoothed=27.5,
        harmonic_evidence_pct=0.92,
        harmonic_evidence_pct_smoothed=0.90,
        best_f0_hz=f0_hz,
        raw_best_f0_hz=f0_hz,
        canonical_best_f0_hz=f0_hz,
        f0_family_stable=True,
        ml_drone_pct=0.96,
        ml_drone_pct_smoothed=0.96,
        combined_drone_evidence_pct=0.93,
        hf_p_drone=0.96,
        hf_negative=False,
        hf_positive=True,
        decision_reason="simulated high acoustic and ML evidence for spatial track separation",
        operator_label="alert",
        candidate_run=4,
        ml_positive_run=4,
        strong_run=3,
        estimated_azimuth_deg=float(bearing_deg) % 360.0,
        beamforming_method="simulated_bearing",
        beam_score=0.82,
        beam_snr_gain_db=8.0,
        beam_confidence_pct=0.82,
        bearing_stable=True,
        bearing_uncertainty_deg=15.0,
        rms=0.04,
        peak=0.45,
        duration_sec=12.0,
        calibrated=True,
        channel_agreement_count=4,
        channel_count=8,
        detector_version="simulate-two-near-one-far-v1",
        station_mode="simulation",
        metadata=metadata,
    )


def build_events(timestamp: float | None = None) -> list[AcousticEvent]:
    timestamp = time.time() if timestamp is None else float(timestamp)
    return [
        build_simulated_event(
            "sim_001",
            latitude=32.10350,
            longitude=34.80800,
            source_id=NEAR_SOURCE_ID,
            coverage_radius_m=250.0,
            bearing_deg=35.0,
            timestamp=timestamp,
        ),
        build_simulated_event(
            "sim_002",
            latitude=32.10420,
            longitude=34.80920,
            source_id=NEAR_SOURCE_ID,
            coverage_radius_m=250.0,
            bearing_deg=305.0,
            timestamp=timestamp,
        ),
        build_simulated_event(
            "sim_003",
            latitude=32.17400,
            longitude=34.90200,
            source_id=FAR_SOURCE_ID,
            coverage_radius_m=150.0,
            bearing_deg=120.0,
            timestamp=timestamp,
        ),
    ]


def build_heartbeat(event: AcousticEvent) -> StationHeartbeat:
    metadata = {
        "source": "simulate_two_near_one_far",
        "scenario_id": SCENARIO_ID,
        "simulated_source_id": (event.metadata or {}).get("simulated_source_id"),
        "coverage_radius_m": (event.metadata or {}).get("coverage_radius_m"),
        "latitude": event.station_latitude,
        "longitude": event.station_longitude,
        "location_label": event.station_location_label,
        "status": "online",
    }
    return StationHeartbeat(
        station_id=event.station_id,
        station_name=event.station_name,
        timestamp_unix=time.time(),
        status="online",
        station_location=event.station_location,
        sample_rate=48000,
        channels=8,
        calibrated=True,
        detector_version="simulate-two-near-one-far-v1",
        station_mode="simulation",
        last_event_status=str(event.status.value if hasattr(event.status, "value") else event.status),
        last_harmonic_score=event.harmonic_score,
        last_hf_p_drone=event.hf_p_drone,
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post a two-near-one-far spatial fusion scenario to SkyEar."
    )
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.set_defaults(heartbeat=True)
    parser.add_argument("--heartbeat", dest="heartbeat", action="store_true")
    parser.add_argument("--no-heartbeat", dest="heartbeat", action="store_false")
    parser.add_argument(
        "--assert-tracks",
        action="store_true",
        help="Fetch /fusion and fail unless sim_001/sim_002 and sim_003 are separate tracks.",
    )
    return parser.parse_args()


def _base_url(events_url: str) -> str:
    return events_url.rsplit("/", 1)[0]


def _track_station_sets(fusion: dict[str, Any]) -> set[tuple[str, ...]]:
    return {tuple(track.get("station_ids") or []) for track in fusion.get("tracks") or []}


def assert_expected_tracks(fusion: dict[str, Any]) -> None:
    station_sets = _track_station_sets(fusion)
    if ("sim_001", "sim_002") not in station_sets:
        raise SystemExit(f"Expected a sim_001/sim_002 track, got {sorted(station_sets)}")
    if ("sim_003",) not in station_sets:
        raise SystemExit(f"Expected a single sim_003 track, got {sorted(station_sets)}")
    if any("sim_003" in station_ids and len(station_ids) > 1 for station_ids in station_sets):
        raise SystemExit(f"sim_003 was incorrectly fused into another track: {sorted(station_sets)}")


def main() -> None:
    args = parse_args()
    heartbeat_url = heartbeat_url_from_events_url(args.server)
    for iteration in range(max(1, int(args.repeat))):
        events = build_events()
        for event in events:
            response = requests.post(args.server, json=event.model_dump(mode="json"), timeout=3.0)
            response.raise_for_status()
            print(
                f"posted {event.station_id} source={(event.metadata or {}).get('simulated_source_id')} "
                f"status={event.status} coverage={(event.metadata or {}).get('coverage_radius_m')}m"
            )
            if args.heartbeat:
                heartbeat = build_heartbeat(event)
                heartbeat_response = requests.post(
                    heartbeat_url,
                    json=heartbeat.model_dump(mode="json"),
                    timeout=3.0,
                )
                heartbeat_response.raise_for_status()
        if iteration + 1 < int(args.repeat):
            time.sleep(max(0.0, float(args.interval_sec)))

    fusion = requests.get(f"{_base_url(args.server)}/fusion", timeout=3.0).json()
    tracks = fusion.get("tracks") or []
    print(
        f"fusion level={fusion.get('level')} interpretation={fusion.get('interpretation')} "
        f"track_count={len(tracks)}"
    )
    for track in tracks:
        print(
            f"{track.get('track_id')} level={track.get('level')} "
            f"confidence={float(track.get('confidence') or 0.0):.2f} "
            f"stations={','.join(track.get('station_ids') or [])} "
            f"same_f0={'yes' if track.get('same_f0') else 'no'}"
        )
    if args.assert_tracks:
        assert_expected_tracks(fusion)


if __name__ == "__main__":
    main()
