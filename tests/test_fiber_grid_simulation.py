from __future__ import annotations

import time

from server.database import InMemoryDatabase
from server.geo_fusion import map_state_from_db
from server.track_fusion import fuse_tracks
from tools.simulate_fiber_grid import (
    generate_fiber_grid_layout,
    simulate_fiber_grid,
)


def test_layout_generator_creates_expected_station_lines():
    stations = generate_fiber_grid_layout(line_spacing_m=80.0, hearing_radius_m=350.0)

    assert len(stations) == 19
    assert [station.station_id for station in stations[:7]] == ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    assert {station.line_distance_m for station in stations if station.line_id == "A"} == {800.0}
    assert {station.line_distance_m for station in stations if station.line_id == "B"} == {600.0}
    assert {station.line_distance_m for station in stations if station.line_id == "C"} == {400.0}
    assert {station.coverage_radius_m for station in stations} == {350.0}


def test_target_near_front_line_triggers_a_line_first():
    simulation = simulate_fiber_grid(steps=1, targets=1, base_time=time.time())
    first_event = simulation.events[0]

    assert first_event.station_id.startswith("A")
    assert first_event.metadata["line_id"] == "A"
    assert simulation.summary[0]["first_detected_line"] == "A"


def test_moving_target_crosses_a_before_b_before_c():
    simulation = simulate_fiber_grid(steps=40, targets=1, speed_mps=18.0, base_time=time.time())
    first_by_line: dict[str, float] = {}
    for event in simulation.events:
        line_id = str(event.metadata["line_id"])
        first_by_line.setdefault(line_id, float(event.metadata["simulation_elapsed_sec"]))

    assert first_by_line["A"] < first_by_line["B"] < first_by_line["C"]
    assert simulation.summary[0]["latest_line_crossed"] == "C"


def test_failed_station_posts_degraded_heartbeat_and_no_events():
    simulation = simulate_fiber_grid(steps=5, targets=1, station_failure="A4", base_time=time.time())

    assert all(event.station_id != "A4" for event in simulation.events)
    failed_heartbeats = [heartbeat for heartbeat in simulation.heartbeats if heartbeat.station_id == "A4"]
    assert failed_heartbeats
    assert all(heartbeat.status == "error" for heartbeat in failed_heartbeats)
    assert all((heartbeat.metadata or {}).get("simulated_station_failure") is True for heartbeat in failed_heartbeats)


def test_one_target_across_multiple_lines_creates_one_track():
    simulation = simulate_fiber_grid(steps=40, targets=1, base_time=time.time())

    fusion = fuse_tracks(simulation.events, window_sec=120.0)

    assert len(fusion.tracks) == 1
    assert {"A", "B", "C"}.issubset({station_id[0] for station_id in fusion.tracks[0].station_ids})


def test_two_separated_targets_create_two_tracks():
    simulation = simulate_fiber_grid(
        steps=40,
        targets=2,
        target_separation_m=300.0,
        base_time=time.time(),
    )

    fusion = fuse_tracks(simulation.events, window_sec=120.0)
    source_sets = []
    for track in fusion.tracks:
        source_sets.append(
            {
                observation.source_hint_id
                for observation in track.observations
                if observation.source_hint_id is not None
            }
        )

    assert len(fusion.tracks) == 2
    assert {"T1"} in source_sets
    assert {"T2"} in source_sets


def test_fiber_grid_map_state_includes_tracks():
    simulation = simulate_fiber_grid(steps=20, targets=1, base_time=time.time())
    db = InMemoryDatabase()
    for event in simulation.events:
        db.add_event(event)
    for heartbeat in simulation.heartbeats[-19:]:
        db.add_heartbeat(heartbeat)

    state = map_state_from_db(db, now=time.time(), fusion_window_sec=120.0)

    assert state["tracks"]
    assert state["tracks"][0]["track_id"]
    assert state["tracks"][0]["source_ids"] == ["T1"]
    assert state["tracks"][0]["target_eta_sec"] is not None
    assert state["tracks"][0]["latest_line_crossed"] in {"A", "B", "C"}
