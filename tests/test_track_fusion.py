from __future__ import annotations

import time

from server.track_fusion import cluster_events_into_tracks, fuse_tracks
from server.track_observations import observations_from_events
from shared.event_schema import AcousticDetectionCandidate, AcousticEvent, GeoPoint
from tools.simulate_two_near_one_far import build_events
from tools.simulate_multi_target import build_events as build_multi_target_events


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


def test_event_with_no_detections_expands_to_one_implicit_observation():
    event = _event("station_1", source_id="source_a", scenario_id="demo")

    observations = observations_from_events([event])

    assert len(observations) == 1
    assert observations[0].station_id == "station_1"
    assert observations[0].source_hint_id == "source_a"
    assert observations[0].candidate_id is None
    assert observations[0].original_event is event


def test_event_with_two_detections_expands_to_two_observations():
    event = _event("station_1", source_id="event_source", scenario_id="demo")
    event.detections = [
        AcousticDetectionCandidate(
            candidate_id="cand_a",
            source_hint_id="source_a",
            confidence=0.88,
            drone_score=0.91,
            bearing_deg=40.0,
            f0_hz=1000,
            harmonic_score=26.0,
        ),
        AcousticDetectionCandidate(
            candidate_id="cand_b",
            source_hint_id="source_b",
            confidence=0.84,
            drone_score=0.89,
            bearing_deg=220.0,
            f0_hz=1320,
            harmonic_score=24.0,
        ),
    ]

    observations = observations_from_events([event])

    assert len(observations) == 2
    assert {observation.candidate_id for observation in observations} == {"cand_a", "cand_b"}
    assert {observation.source_hint_id for observation in observations} == {"source_a", "source_b"}


def test_same_station_two_detections_can_contribute_to_two_tracks():
    timestamp = time.time()
    station_a = _event("station_A1", latitude=32.0, longitude=34.0, source_id="target_1", scenario_id="multi")
    station_a.timestamp_unix = timestamp
    station_a.server_received_unix = timestamp
    station_b = _event("station_A2", latitude=32.0, longitude=34.001, source_id=None, scenario_id="multi")
    station_b.timestamp_unix = timestamp
    station_b.server_received_unix = timestamp
    station_b.detections = [
        AcousticDetectionCandidate(
            candidate_id="A2_T1",
            source_hint_id="target_1",
            confidence=0.92,
            drone_score=0.93,
            bearing_deg=35.0,
            f0_hz=1020,
            harmonic_score=27.0,
            metadata={"scenario_id": "multi", "simulated_source_id": "target_1"},
        ),
        AcousticDetectionCandidate(
            candidate_id="A2_T2",
            source_hint_id="target_2",
            confidence=0.90,
            drone_score=0.91,
            bearing_deg=135.0,
            f0_hz=1320,
            harmonic_score=26.0,
            metadata={"scenario_id": "multi", "simulated_source_id": "target_2"},
        ),
    ]
    station_c = _event("station_A3", latitude=32.0, longitude=34.002, source_id="target_2", scenario_id="multi")
    station_c.timestamp_unix = timestamp
    station_c.server_received_unix = timestamp

    tracks = cluster_events_into_tracks([station_a, station_b, station_c])
    station_sets = {tuple(track.station_ids) for track in tracks}

    assert len(tracks) == 2
    assert ("station_A1", "station_A2") in station_sets
    assert ("station_A2", "station_A3") in station_sets
    assert sum("station_A2" in track.station_ids for track in tracks) == 2
    assert any(len(track.observations) == 2 for track in tracks)


def test_ambiguous_same_station_same_sector_is_not_reported_as_precise_split():
    event = _event("station_1")
    event.detections = [
        AcousticDetectionCandidate(
            candidate_id="cand_a",
            confidence=0.86,
            drone_score=0.88,
            bearing_deg=60.0,
            f0_hz=1000,
            harmonic_score=24.0,
            metadata={"sector_id": "north_east"},
        ),
        AcousticDetectionCandidate(
            candidate_id="cand_b",
            confidence=0.84,
            drone_score=0.86,
            bearing_deg=64.0,
            f0_hz=1060,
            harmonic_score=23.0,
            metadata={"sector_id": "north_east"},
        ),
    ]

    tracks = cluster_events_into_tracks([event])

    assert len(tracks) == 1
    assert tracks[0].target_count_hint == 2
    assert tracks[0].ambiguity == "possible split acoustic source; possible 1-2 targets"


def test_multi_target_simulation_returns_exactly_two_tracks():
    fusion = fuse_tracks(build_multi_target_events(targets=2, stations=4))
    station_sets = {tuple(track.station_ids) for track in fusion.tracks}

    assert len(fusion.tracks) == 2
    assert ("sim_A1", "sim_A2") in station_sets
    assert ("sim_A2", "sim_A3", "sim_A4") in station_sets
    assert sum("sim_A2" in track.station_ids for track in fusion.tracks) == 2
