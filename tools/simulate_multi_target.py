from __future__ import annotations

import argparse
import time
from typing import Any

import requests

from shared.event_schema import AcousticDetectionCandidate, AcousticEvent, EventStatus, GeoPoint, StationHeartbeat
from station.station_agent import heartbeat_url_from_events_url


SCENARIO_ID = "multi_target"


def _metadata(source_id: str, *, source: str = "simulate_multi_target") -> dict[str, Any]:
    return {
        "source": source,
        "scenario_id": SCENARIO_ID,
        "simulated_source_id": source_id,
        "coverage_radius_m": 300.0,
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
        "note": "synthetic multi-target fusion test; passive warning/demo data only",
    }


def build_simulated_event(
    station_id: str,
    *,
    latitude: float,
    longitude: float,
    source_id: str,
    bearing_deg: float,
    f0_hz: int,
    timestamp: float | None = None,
    detections: list[AcousticDetectionCandidate] | None = None,
) -> AcousticEvent:
    timestamp = time.time() if timestamp is None else float(timestamp)
    return AcousticEvent(
        station_id=station_id,
        station_name=f"Simulated {station_id}",
        timestamp_unix=timestamp,
        station_location=GeoPoint(latitude=latitude, longitude=longitude, altitude_m=0.0),
        station_latitude=latitude,
        station_longitude=longitude,
        station_altitude_m=0.0,
        station_location_label="multi-target simulation",
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
        decision_reason="simulated multi-target acoustic evidence",
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
        detector_version="simulate-multi-target-v1",
        station_mode="simulation",
        detections=detections or [],
        metadata=_metadata(source_id),
    )


def build_events(*, targets: int = 2, stations: int = 4, timestamp: float | None = None) -> list[AcousticEvent]:
    if int(targets) != 2 or int(stations) != 4:
        raise ValueError("The phase-1 multi-target simulation currently supports --targets 2 --stations 4.")
    timestamp = time.time() if timestamp is None else float(timestamp)
    shared_station_detections = [
        AcousticDetectionCandidate(
            candidate_id="A2_T1",
            source_hint_id="T1",
            confidence=0.92,
            drone_score=0.93,
            bearing_deg=35.0,
            bearing_error_deg=15.0,
            bearing_quality="good",
            f0_hz=1040,
            harmonic_score=27.0,
            power=0.72,
            metadata=_metadata("T1"),
        ),
        AcousticDetectionCandidate(
            candidate_id="A2_T2",
            source_hint_id="T2",
            confidence=0.90,
            drone_score=0.91,
            bearing_deg=145.0,
            bearing_error_deg=18.0,
            bearing_quality="good",
            f0_hz=1320,
            harmonic_score=26.0,
            power=0.68,
            metadata=_metadata("T2"),
        ),
    ]
    return [
        build_simulated_event(
            "sim_A1",
            latitude=32.10350,
            longitude=34.80800,
            source_id="T1",
            bearing_deg=35.0,
            f0_hz=1040,
            timestamp=timestamp,
        ),
        build_simulated_event(
            "sim_A2",
            latitude=32.10420,
            longitude=34.80920,
            source_id="T1",
            bearing_deg=60.0,
            f0_hz=1040,
            timestamp=timestamp,
            detections=shared_station_detections,
        ),
        build_simulated_event(
            "sim_A3",
            latitude=32.10620,
            longitude=34.81220,
            source_id="T2",
            bearing_deg=320.0,
            f0_hz=1320,
            timestamp=timestamp,
        ),
        build_simulated_event(
            "sim_A4",
            latitude=32.10720,
            longitude=34.81340,
            source_id="T2",
            bearing_deg=290.0,
            f0_hz=1320,
            timestamp=timestamp,
        ),
    ]


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
        detector_version="simulate-multi-target-v1",
        station_mode="simulation",
        last_event_status=str(event.status.value if hasattr(event.status, "value") else event.status),
        last_harmonic_score=event.harmonic_score,
        last_hf_p_drone=event.hf_p_drone,
        metadata={
            "source": "simulate_multi_target",
            "scenario_id": SCENARIO_ID,
            "latitude": event.station_latitude,
            "longitude": event.station_longitude,
            "location_label": event.station_location_label,
            "status": "online",
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post a multi-target spatial fusion scenario to SkyEar.")
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--stations", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.set_defaults(heartbeat=True)
    parser.add_argument("--heartbeat", dest="heartbeat", action="store_true")
    parser.add_argument("--no-heartbeat", dest="heartbeat", action="store_false")
    parser.add_argument("--assert-tracks", action="store_true")
    return parser.parse_args(argv)


def _base_url(events_url: str) -> str:
    return events_url.rsplit("/", 1)[0]


def _track_station_sets(fusion: dict[str, Any]) -> set[tuple[str, ...]]:
    return {tuple(track.get("station_ids") or []) for track in fusion.get("tracks") or []}


def assert_expected_tracks(fusion: dict[str, Any]) -> None:
    station_sets = _track_station_sets(fusion)
    expected = {("sim_A1", "sim_A2"), ("sim_A2", "sim_A3", "sim_A4")}
    if not expected.issubset(station_sets):
        raise SystemExit(f"Expected multi-target station sets {sorted(expected)}, got {sorted(station_sets)}")
    if sum("sim_A2" in station_ids for station_ids in station_sets) < 2:
        raise SystemExit(f"Expected sim_A2 to contribute to two tracks, got {sorted(station_sets)}")
    if len(fusion.get("tracks") or []) != 2:
        raise SystemExit(f"Expected exactly 2 tracks, got {len(fusion.get('tracks') or [])}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    heartbeat_url = heartbeat_url_from_events_url(args.server)
    for iteration in range(max(1, int(args.repeat))):
        for event in build_events(targets=args.targets, stations=args.stations):
            response = requests.post(args.server, json=event.model_dump(mode="json"), timeout=3.0)
            response.raise_for_status()
            print(
                f"posted {event.station_id} detections={len(event.detections)} "
                f"source={(event.metadata or {}).get('simulated_source_id')}"
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
            f"observations={len(track.get('observations') or [])}"
        )
    if args.assert_tracks:
        assert_expected_tracks(fusion)


if __name__ == "__main__":
    main()
