from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any

import requests

from server.geo import destination_point, haversine_distance_m
from shared.event_schema import AcousticEvent, EventStatus, GeoPoint, StationHeartbeat
from station.station_agent import heartbeat_url_from_events_url


SCENARIO_ID = "fiber_grid"
DEFAULT_CONTROL_LAT = 32.0853
DEFAULT_CONTROL_LON = 34.7818
LINE_SPECS = (("A", 800.0, 7), ("B", 600.0, 5), ("C", 400.0, 7))


@dataclass(frozen=True)
class FiberStation:
    station_id: str
    line_id: str
    line_distance_m: float
    fiber_node_id: str
    station_group: str
    coverage_radius_m: float
    latitude: float
    longitude: float
    health: str = "online"

    def metadata(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "line_id": self.line_id,
            "line_distance_m": self.line_distance_m,
            "fiber_node_id": self.fiber_node_id,
            "station_group": self.station_group,
            "coverage_radius_m": self.coverage_radius_m,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "health": self.health,
        }


@dataclass(frozen=True)
class SimTarget:
    source_id: str
    f0_hz: int
    speed_mps: float
    heading_deg: float
    lateral_offset_m: float
    start_distance_m: float


@dataclass
class FiberGridSimulation:
    stations: list[FiberStation]
    targets: list[SimTarget]
    events: list[AcousticEvent]
    heartbeats: list[StationHeartbeat]
    summary: list[dict[str, Any]]


def _offset_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> dict[str, float]:
    if distance_m < 0:
        return destination_point(lat, lon, (float(bearing_deg) + 180.0) % 360.0, abs(float(distance_m)))
    return destination_point(lat, lon, float(bearing_deg), float(distance_m))


def _approach_bearing(target_heading_deg: float) -> float:
    return (float(target_heading_deg) + 180.0) % 360.0


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_lon = math.radians(float(lon2) - float(lon1))
    y = math.sin(delta_lon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def parse_station_failures(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def generate_fiber_grid_layout(
    *,
    control_lat: float = DEFAULT_CONTROL_LAT,
    control_lon: float = DEFAULT_CONTROL_LON,
    target_heading_deg: float = 180.0,
    line_spacing_m: float = 80.0,
    hearing_radius_m: float = 350.0,
    failed_stations: set[str] | None = None,
) -> list[FiberStation]:
    failed_stations = failed_stations or set()
    approach = _approach_bearing(target_heading_deg)
    lateral_heading = (approach + 90.0) % 360.0
    stations: list[FiberStation] = []
    for line_id, line_distance_m, count in LINE_SPECS:
        center = destination_point(control_lat, control_lon, approach, line_distance_m)
        center_offset = (count - 1) / 2.0
        for idx in range(count):
            offset_m = (idx - center_offset) * float(line_spacing_m)
            point = _offset_point(center["latitude"], center["longitude"], lateral_heading, offset_m)
            station_id = f"{line_id}{idx + 1}"
            stations.append(
                FiberStation(
                    station_id=station_id,
                    line_id=line_id,
                    line_distance_m=float(line_distance_m),
                    fiber_node_id=f"fiber_{line_id}",
                    station_group=f"line_{line_id}",
                    coverage_radius_m=float(hearing_radius_m),
                    latitude=point["latitude"],
                    longitude=point["longitude"],
                    health="offline" if station_id in failed_stations else "online",
                )
            )
    return stations


def build_targets(
    *,
    targets: int = 1,
    target_heading_deg: float = 180.0,
    speed_mps: float = 18.0,
    hearing_radius_m: float = 350.0,
    target_separation_m: float = 300.0,
) -> list[SimTarget]:
    count = max(1, int(targets))
    if count > 2:
        raise ValueError("fiber-grid simulation currently supports --targets 1 or 2")
    offsets = [0.0] if count == 1 else [-float(target_separation_m) / 2.0, float(target_separation_m) / 2.0]
    f0s = [1205, 950]
    start_distance = 800.0 + float(hearing_radius_m) * 0.50
    return [
        SimTarget(
            source_id=f"T{idx + 1}",
            f0_hz=f0s[idx],
            speed_mps=float(speed_mps),
            heading_deg=float(target_heading_deg),
            lateral_offset_m=float(offsets[idx]),
            start_distance_m=float(start_distance),
        )
        for idx in range(count)
    ]


def target_position(
    *,
    control_lat: float,
    control_lon: float,
    target: SimTarget,
    elapsed_sec: float,
) -> dict[str, float]:
    approach = _approach_bearing(target.heading_deg)
    radial_distance = max(0.0, float(target.start_distance_m) - float(target.speed_mps) * float(elapsed_sec))
    radial = destination_point(control_lat, control_lon, approach, radial_distance)
    lateral_heading = (approach + 90.0) % 360.0
    point = _offset_point(radial["latitude"], radial["longitude"], lateral_heading, target.lateral_offset_m)
    point["distance_to_control_m"] = radial_distance
    return point


def _line_crossed(distance_to_control_m: float) -> str | None:
    crossed = [line_id for line_id, line_distance_m, _count in LINE_SPECS if float(distance_to_control_m) <= line_distance_m]
    return crossed[-1] if crossed else None


def _deterministic_noise_deg(station_id: str, source_id: str, step_index: int, scale_deg: float) -> float:
    seed = sum(ord(ch) for ch in f"{station_id}:{source_id}") + int(step_index) * 17
    return math.sin(seed * 0.73) * float(scale_deg)


def build_fiber_event(
    station: FiberStation,
    target: SimTarget,
    *,
    target_latitude: float,
    target_longitude: float,
    distance_m: float,
    target_distance_to_control_m: float,
    timestamp: float,
    elapsed_sec: float,
    step_index: int,
    bearing_noise_deg: float = 2.0,
) -> AcousticEvent:
    true_bearing = _bearing_deg(station.latitude, station.longitude, target_latitude, target_longitude)
    bearing = (true_bearing + _deterministic_noise_deg(station.station_id, target.source_id, step_index, bearing_noise_deg)) % 360.0
    distance_ratio = max(0.0, min(1.0, 1.0 - float(distance_m) / max(1.0, station.coverage_radius_m)))
    confidence = max(0.20, min(0.96, 0.36 + 0.58 * distance_ratio))
    harmonic_score = 16.0 + 17.0 * distance_ratio
    hf_p_drone = max(0.45, min(0.98, 0.64 + 0.32 * distance_ratio))
    combined = max(0.35, min(0.96, 0.50 + 0.43 * distance_ratio))
    uncertainty = 12.0 + 34.0 * (1.0 - distance_ratio)
    target_eta_sec = None if target.speed_mps <= 0 else target_distance_to_control_m / target.speed_mps
    status = EventStatus.ALERT if combined >= 0.75 else EventStatus.DRONE_LIKE if combined >= 0.60 else EventStatus.SUSPECT
    metadata = {
        **station.metadata(),
        "source": "simulate_fiber_grid",
        "scenario_id": SCENARIO_ID,
        "simulated_source_id": target.source_id,
        "true_source_latitude": float(target_latitude),
        "true_source_longitude": float(target_longitude),
        "true_source_bearing_deg": true_bearing,
        "true_distance_m": float(distance_m),
        "target_speed_mps": float(target.speed_mps),
        "target_heading_deg": float(target.heading_deg),
        "target_distance_to_control_m": float(target_distance_to_control_m),
        "target_eta_sec": target_eta_sec,
        "simulation_step": int(step_index),
        "simulation_elapsed_sec": float(elapsed_sec),
        "latest_line_crossed": _line_crossed(target_distance_to_control_m),
        "harmonic_evidence_pct": combined,
        "harmonic_evidence_pct_smoothed": combined,
        "ml_drone_pct": hf_p_drone,
        "combined_drone_evidence_pct": combined,
        "f0_stable": True,
        "f0_family_stable": True,
        "candidate_run": 4,
        "ml_positive_run": 4,
        "strong_run": 3,
        "suspect_threshold": 16.0,
        "alert_threshold": 22.0,
        "bearing_deg": bearing,
        "bearing_uncertainty_deg": uncertainty,
        "bearing_reliable": True,
        "bearing_used_for_geo": True,
        "bearing_quality": "good" if uncertainty <= 24.0 else "fair",
        "operator_label": "local_drone_candidate",
    }
    return AcousticEvent(
        station_id=station.station_id,
        station_name=f"Fiber {station.station_id}",
        timestamp_unix=float(timestamp),
        server_received_unix=time.time(),
        station_location=GeoPoint(latitude=station.latitude, longitude=station.longitude, altitude_m=0.0),
        station_latitude=station.latitude,
        station_longitude=station.longitude,
        station_altitude_m=0.0,
        station_location_label=f"fiber line {station.line_id}",
        status=status,
        confidence=confidence,
        harmonic_score=harmonic_score,
        harmonic_score_smoothed=harmonic_score,
        harmonic_evidence_pct=combined,
        harmonic_evidence_pct_smoothed=combined,
        best_f0_hz=target.f0_hz,
        raw_best_f0_hz=target.f0_hz,
        canonical_best_f0_hz=target.f0_hz,
        f0_family_stable=True,
        ml_drone_pct=hf_p_drone,
        ml_drone_pct_smoothed=hf_p_drone,
        combined_drone_evidence_pct=combined,
        hf_p_drone=hf_p_drone,
        hf_negative=False,
        hf_positive=True,
        decision_reason="synthetic fiber-grid moving target",
        operator_label="local_drone_candidate",
        candidate_run=4,
        ml_positive_run=4,
        strong_run=3,
        estimated_azimuth_deg=bearing,
        raw_bearing_deg=bearing,
        tracked_bearing_deg=bearing,
        bearing_track_status="tracking",
        bearing_track_stable=True,
        beamforming_method="simulated_bearing",
        beam_score=confidence,
        beam_snr_gain_db=8.0 * distance_ratio,
        beam_confidence_pct=confidence,
        bearing_stable=True,
        bearing_uncertainty_deg=uncertainty,
        bearing_reliable=True,
        bearing_quality=metadata["bearing_quality"],
        bearing_used_for_geo=True,
        rms=0.01 + 0.05 * distance_ratio,
        peak=0.15 + 0.45 * distance_ratio,
        duration_sec=1.0,
        calibrated=True,
        channel_agreement_count=4,
        channel_count=8,
        detector_version="simulate-fiber-grid-v1",
        station_mode="simulation",
        metadata=metadata,
    )


def build_fiber_heartbeat(station: FiberStation, *, timestamp: float, failed: bool = False) -> StationHeartbeat:
    status = "error" if failed else "online"
    metadata = {
        **station.metadata(),
        "source": "simulate_fiber_grid",
        "scenario_id": SCENARIO_ID,
        "fiber_connected": not failed,
        "station_power_state": "offline" if failed else "online",
        "simulated_station_failure": bool(failed),
        "status": status,
    }
    return StationHeartbeat(
        station_id=station.station_id,
        station_name=f"Fiber {station.station_id}",
        timestamp_unix=float(timestamp),
        server_received_unix=time.time(),
        status=status,
        station_location=GeoPoint(latitude=station.latitude, longitude=station.longitude, altitude_m=0.0),
        sample_rate=48000,
        channels=8,
        calibrated=True,
        detector_version="simulate-fiber-grid-v1",
        station_mode="simulation",
        last_error="simulated station failure" if failed else None,
        metadata=metadata,
    )


def simulate_fiber_grid(
    *,
    control_lat: float = DEFAULT_CONTROL_LAT,
    control_lon: float = DEFAULT_CONTROL_LON,
    targets: int = 1,
    steps: int = 60,
    step_sec: float = 1.0,
    speed_mps: float = 18.0,
    hearing_radius_m: float = 350.0,
    line_spacing_m: float = 80.0,
    target_heading_deg: float = 180.0,
    station_failure: str | None = None,
    target_separation_m: float = 300.0,
    bearing_noise_deg: float = 2.0,
    base_time: float | None = None,
) -> FiberGridSimulation:
    failed = parse_station_failures(station_failure)
    stations = generate_fiber_grid_layout(
        control_lat=control_lat,
        control_lon=control_lon,
        target_heading_deg=target_heading_deg,
        line_spacing_m=line_spacing_m,
        hearing_radius_m=hearing_radius_m,
        failed_stations=failed,
    )
    sim_targets = build_targets(
        targets=targets,
        target_heading_deg=target_heading_deg,
        speed_mps=speed_mps,
        hearing_radius_m=hearing_radius_m,
        target_separation_m=target_separation_m,
    )
    base_time = time.time() if base_time is None else float(base_time)
    events: list[AcousticEvent] = []
    heartbeats: list[StationHeartbeat] = []
    summary: dict[str, dict[str, Any]] = {
        target.source_id: {
            "source_id": target.source_id,
            "first_detected_line": None,
            "first_station": None,
            "first_detection_elapsed_sec": None,
            "latest_line_crossed": None,
            "estimated_eta_sec": None,
            "actual_eta_to_control_sec": None if target.speed_mps <= 0 else target.start_distance_m / target.speed_mps,
        }
        for target in sim_targets
    }

    for step_index in range(max(1, int(steps))):
        elapsed_sec = step_index * float(step_sec)
        timestamp = base_time + elapsed_sec
        for station in stations:
            heartbeats.append(build_fiber_heartbeat(station, timestamp=timestamp, failed=station.station_id in failed))
        for target in sim_targets:
            point = target_position(
                control_lat=control_lat,
                control_lon=control_lon,
                target=target,
                elapsed_sec=elapsed_sec,
            )
            target_distance_to_control = float(point["distance_to_control_m"])
            crossed = _line_crossed(target_distance_to_control)
            if crossed is not None:
                summary[target.source_id]["latest_line_crossed"] = crossed
            summary[target.source_id]["estimated_eta_sec"] = (
                None if target.speed_mps <= 0 else target_distance_to_control / target.speed_mps
            )
            for station in stations:
                if station.station_id in failed:
                    continue
                distance = haversine_distance_m(
                    station.latitude,
                    station.longitude,
                    point["latitude"],
                    point["longitude"],
                )
                if distance > float(hearing_radius_m):
                    continue
                event = build_fiber_event(
                    station,
                    target,
                    target_latitude=point["latitude"],
                    target_longitude=point["longitude"],
                    distance_m=distance,
                    target_distance_to_control_m=target_distance_to_control,
                    timestamp=timestamp,
                    elapsed_sec=elapsed_sec,
                    step_index=step_index,
                    bearing_noise_deg=bearing_noise_deg,
                )
                events.append(event)
                target_summary = summary[target.source_id]
                if target_summary["first_detected_line"] is None:
                    target_summary["first_detected_line"] = station.line_id
                    target_summary["first_station"] = station.station_id
                    target_summary["first_detection_elapsed_sec"] = elapsed_sec

    return FiberGridSimulation(
        stations=stations,
        targets=sim_targets,
        events=events,
        heartbeats=heartbeats,
        summary=list(summary.values()),
    )


def _base_url(events_url: str) -> str:
    return events_url.rsplit("/", 1)[0]


def _detected_source_ids(events: list[AcousticEvent]) -> set[str]:
    return {
        str((event.metadata or {}).get("simulated_source_id"))
        for event in events
        if (event.metadata or {}).get("simulated_source_id") is not None
    }


def assert_expected_tracks(fusion: dict[str, Any], *, detected_source_ids: set[str]) -> None:
    tracks = fusion.get("tracks") or []
    if len(tracks) < len(detected_source_ids):
        raise SystemExit(f"Expected at least {len(detected_source_ids)} track(s), got {len(tracks)}")
    covered: set[str] = set()
    for track in tracks:
        track_sources: set[str] = set()
        for observation in track.get("observations") or []:
            source_id = observation.get("source_hint_id")
            if source_id is None:
                source_id = (observation.get("metadata") or {}).get("simulated_source_id")
            if source_id is not None:
                track_sources.add(str(source_id))
        covered |= track_sources
    missing = detected_source_ids - covered
    if missing:
        raise SystemExit(f"Fusion tracks did not cover detected source(s): {sorted(missing)}")


def print_summary(simulation: FiberGridSimulation) -> None:
    print(
        f"layout stations={len(simulation.stations)} events={len(simulation.events)} "
        f"heartbeats={len(simulation.heartbeats)} targets={len(simulation.targets)}"
    )
    for item in simulation.summary:
        print(
            f"{item['source_id']}: first_line={item['first_detected_line']} "
            f"first_station={item['first_station']} "
            f"first_t={item['first_detection_elapsed_sec']}s "
            f"latest_line={item['latest_line_crossed']} "
            f"eta_est={None if item['estimated_eta_sec'] is None else round(float(item['estimated_eta_sec']), 1)}s "
            f"eta_actual={None if item['actual_eta_to_control_sec'] is None else round(float(item['actual_eta_to_control_sec']), 1)}s"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post a synthetic fiber-grid / defense-line scenario to SkyEar.")
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--control-lat", type=float, default=DEFAULT_CONTROL_LAT)
    parser.add_argument("--control-lon", type=float, default=DEFAULT_CONTROL_LON)
    parser.add_argument("--target-heading-deg", type=float, default=180.0)
    parser.add_argument("--targets", type=int, choices=[1, 2], default=1)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--speed-mps", type=float, default=18.0)
    parser.add_argument("--hearing-radius-m", type=float, default=350.0)
    parser.add_argument("--line-spacing-m", type=float, default=80.0)
    parser.add_argument("--target-separation-m", type=float, default=300.0)
    parser.add_argument("--station-failure", default="")
    parser.add_argument("--bearing-noise-deg", type=float, default=2.0)
    parser.set_defaults(post_realtime=False, heartbeat=True)
    parser.add_argument("--post-realtime", dest="post_realtime", action="store_true")
    parser.add_argument("--no-realtime", dest="post_realtime", action="store_false")
    parser.add_argument("--heartbeat", dest="heartbeat", action="store_true")
    parser.add_argument("--no-heartbeat", dest="heartbeat", action="store_false")
    parser.add_argument("--assert-tracks", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    simulation = simulate_fiber_grid(
        control_lat=args.control_lat,
        control_lon=args.control_lon,
        targets=args.targets,
        steps=args.steps,
        step_sec=args.step_sec,
        speed_mps=args.speed_mps,
        hearing_radius_m=args.hearing_radius_m,
        line_spacing_m=args.line_spacing_m,
        target_heading_deg=args.target_heading_deg,
        station_failure=args.station_failure,
        target_separation_m=args.target_separation_m,
        bearing_noise_deg=args.bearing_noise_deg,
    )
    print_summary(simulation)
    heartbeat_url = heartbeat_url_from_events_url(args.server)
    heartbeat_by_time: dict[float, list[StationHeartbeat]] = {}
    for heartbeat in simulation.heartbeats:
        heartbeat_by_time.setdefault(float(heartbeat.timestamp_unix), []).append(heartbeat)
    events_by_time: dict[float, list[AcousticEvent]] = {}
    for event in simulation.events:
        events_by_time.setdefault(float(event.timestamp_unix), []).append(event)

    timestamps = sorted(set(heartbeat_by_time) | set(events_by_time))
    for idx, timestamp in enumerate(timestamps):
        if args.heartbeat:
            for heartbeat in heartbeat_by_time.get(timestamp, []):
                response = requests.post(heartbeat_url, json=heartbeat.model_dump(mode="json"), timeout=3.0)
                response.raise_for_status()
        for event in events_by_time.get(timestamp, []):
            response = requests.post(args.server, json=event.model_dump(mode="json"), timeout=3.0)
            response.raise_for_status()
        if events_by_time.get(timestamp):
            print(f"posted step={idx} events={len(events_by_time[timestamp])}")
        if args.post_realtime and idx + 1 < len(timestamps):
            delay = max(0.0, min(float(args.step_sec), timestamps[idx + 1] - timestamp))
            time.sleep(delay)

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
            f"stations={','.join(track.get('station_ids') or [])}"
        )
    if args.assert_tracks:
        assert_expected_tracks(fusion, detected_source_ids=_detected_source_ids(simulation.events))


if __name__ == "__main__":
    main()
