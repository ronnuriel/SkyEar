from __future__ import annotations

from station.detector_state import StationDetectorStateConfig
from tools.simulate_client_demo import demo_phase
from tools.simulate_station import build_event, generate_synthetic_audio, make_simulated_stations


SR = 16000
WINDOW_SEC = 0.5


def _stations(count: int = 3):
    cfg = StationDetectorStateConfig(
        f0_min=500,
        f0_max=1500,
        max_freq=5000,
        calibration_seconds=1.0,
        min_alert_duration_sec=1.0,
        clear_after_sec=1.0,
    )
    stations = make_simulated_stations(count, "sim_001", cfg)
    for station in stations:
        for ts in (0.0, 1.0):
            audio = generate_synthetic_audio("background", ts, SR, WINDOW_SEC, 8, station.station_index)
            build_event(station, audio, SR, ts, max_freq=5000)
    return stations


def test_motorcycle_like_does_not_produce_alert_after_calibration():
    station = _stations(1)[0]

    audio = generate_synthetic_audio("motorcycle_like", 10.0, SR, WINDOW_SEC, 8)
    event = build_event(station, audio, SR, 2.0, max_freq=5000)

    assert event.status != "alert"


def test_two_station_drone_produces_suspect_or_alert_events():
    stations = _stations(2)
    statuses = []

    for station in stations:
        build_event(
            station,
            generate_synthetic_audio("multi_rotor_jitter", 10.0, SR, WINDOW_SEC, 8, station.station_index),
            SR,
            2.0,
            max_freq=5000,
        )
        event = build_event(
            station,
            generate_synthetic_audio("multi_rotor_jitter", 11.0, SR, WINDOW_SEC, 8, station.station_index),
            SR,
            3.1,
            max_freq=5000,
        )
        statuses.append(event.status.value)

    assert all(status in {"suspect", "drone_like", "alert"} for status in statuses)


def test_single_station_demo_phase_only_affects_one_station():
    phase, scenarios = demo_phase(40.0)

    assert phase == "single_station_drone"
    assert scenarios == ["multi_rotor_jitter", "background", "background"]
