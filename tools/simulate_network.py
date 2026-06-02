from __future__ import annotations

import argparse

from tools.simulate_station import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--stations", type=int, default=3)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument(
        "--scenario",
        choices=[
            "background",
            "drone_hover",
            "drone_pass",
            "multi_rotor_jitter",
            "false_positive_fan",
            "motorcycle_like",
        ],
        default="drone_pass",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--heartbeat", action="store_true")
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    args = parser.parse_args()
    args.num_stations = args.stations
    args.station_id = "sim_001"
    return args


def main() -> None:
    run_simulation(parse_args())


if __name__ == "__main__":
    main()
