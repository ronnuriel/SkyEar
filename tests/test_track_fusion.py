from __future__ import annotations

import time

from server.track_fusion import cluster_events_into_tracks, fuse_tracks
from shared.event_schema import AcousticEvent, GeoPoint
from tools.simulate_two_near_one_far import build_events


def _event(
    station_id: str,
    *,
    latitude: float = 32.0,
    longitude: float = 34.0,
    coverage_radius_m: float = 150.0,
    best_f0_hz: int = 1000,
    combined: float = 0.62,
    source_id: str | None = None,
    scenario_id: str | None = None,
) -> AcousticEvent:
    metadata = {
        "coverage_radius_m": coverage_radius_m,
        "combined_drone_evidence_pct": combined,
        "harmonic_evidence_pct": 0.45,
        "ml_drone_pct": 1.0,
        "suspect_threshold": 16.0,
        "alert_threshold": 22.0,
    }
    if scenario_id is not None:
        metadata["scenario_id"] = scenario_id
    if source_id is not None:
        metadata["simulated_source_id"] = source_id
    return AcousticEvent(
        station_id=station_id,
        timestamp_unix=time.time(),
        server_received_unix=time.time(),
        station_location=GeoPoint(latitude=latitude, longitude=longitude),
        status="suspect",
        confidence=0.2,
        harmonic_score=18.0,
        harmonic_evidence_pct=0.45,
        ml_drone_pct=1.0,
        combined_drone_evidence_pct=combined,
        hf_p_drone=1.0,
        best_f0_hz=best_f0_hz,
        calibrated=True,
        channel_agreement_count=1,
        channel_count=1,
        metadata=metadata,
    )


def test_two_close_stations_with_same_f0_cluster_into_one_track():
    events = [
        _event("station_1", latitude=32.0, longitude=34.0, best_f0_hz=1000),
        _event("station_2", latitude=32.0, longitude=34.001, best_f0_hz=1040),
    ]

    tracks = cluster_events_into_tracks(events)

    assert len(tracks) == 1
    assert tracks[0].station_ids == ["station_1", "station_2"]
    assert tracks[0].same_f0 is True


def test_two_far_stations_do_not_cluster_even_with_same_f0():
    events = [
        _event("station_1", latitude=32.0, longitude=34.0, best_f0_hz=1000),
        _event("station_2", latitude=32.0, longitude=34.1, best_f0_hz=1040),
    ]

    tracks = cluster_events_into_tracks(events)

    assert len(tracks) == 2
    assert {tuple(track.station_ids) for track in tracks} == {("station_1",), ("station_2",)}


def test_far_stations_with_different_simulated_source_ids_create_two_tracks():
    events = [
        _event("station_1", latitude=32.0, longitude=34.0, scenario_id="demo", source_id="source_a"),
        _event("station_2", latitude=32.0, longitude=34.1, scenario_id="demo", source_id="source_b"),
    ]

    tracks = cluster_events_into_tracks(events)

    assert len(tracks) == 2


def test_same_simulated_source_id_clusters_even_when_far():
    events = [
        _event("station_1", latitude=32.0, longitude=34.0, scenario_id="demo", source_id="source_a"),
        _event("station_2", latitude=32.0, longitude=34.1, scenario_id="demo", source_id="source_a"),
    ]

    tracks = cluster_events_into_tracks(events)

    assert len(tracks) == 1
    assert tracks[0].station_ids == ["station_1", "station_2"]


def test_global_level_is_max_track_level_not_sum_of_unrelated_tracks():
    events = [
        _event("station_1", latitude=32.0, longitude=34.0, combined=0.62),
        _event("station_2", latitude=32.0, longitude=34.1, combined=0.62),
    ]

    fusion = fuse_tracks(events)

    assert len(fusion.tracks) == 2
    assert [track.level for track in fusion.tracks] == [1, 1]
    assert fusion.global_level == 1
    assert fusion.level == 1
    assert fusion.interpretation == "multiple local candidates"


def test_two_near_one_far_simulation_creates_two_tracks():
    tracks = cluster_events_into_tracks(build_events())
    station_sets = {tuple(track.station_ids) for track in tracks}

    assert len(tracks) >= 2
    assert ("sim_001", "sim_002") in station_sets
    assert ("sim_003",) in station_sets
    assert all("sim_003" not in track.station_ids or len(track.station_ids) == 1 for track in tracks)

    near_track = next(track for track in tracks if track.station_ids == ["sim_001", "sim_002"])
    far_track = next(track for track in tracks if track.station_ids == ["sim_003"])

    assert near_track.level >= 2
    assert far_track.level >= 1
    assert near_track.interpretation == "multi-station overlapping candidate"
    assert far_track.interpretation == "single-station candidate"


def test_two_near_one_far_fusion_does_not_add_far_station_to_near_track():
    fusion = fuse_tracks(build_events())

    assert len(fusion.tracks) >= 2
    assert fusion.interpretation == "multiple local candidates"
    near_track = next(track for track in fusion.tracks if track.station_ids == ["sim_001", "sim_002"])

    assert near_track.station_ids == ["sim_001", "sim_002"]
    assert "sim_003" not in near_track.station_ids
    assert "sim_003" not in near_track.reason
