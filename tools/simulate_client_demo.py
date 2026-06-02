from __future__ import annotations

import argparse
import time

import requests

from tools.simulate_station import build_event, generate_synthetic_audio, make_simulated_stations


def demo_phase(elapsed_sec: float) -> tuple[str, list[str]]:
    if elapsed_sec < 8.0:
        return "background", ["background", "background", "background"]
    if elapsed_sec < 18.0:
        return "motorcycle_false_positive_test", ["motorcycle_like", "motorcycle_like", "background"]
    if elapsed_sec < 34.0:
        return "two_station_drone", ["multi_rotor_jitter", "multi_rotor_jitter", "background"]
    if elapsed_sec < 48.0:
        return "single_station_drone", ["multi_rotor_jitter", "background", "background"]
    return "all_clear", ["background", "background", "background"]


def run_demo(args: argparse.Namespace) -> None:
    stations = make_simulated_stations(3, "sim_001")
    start = time.time()
    total_duration = 60.0

    while True:
        loop_start = time.time()
        elapsed = loop_start - start
        phase, scenarios = demo_phase(elapsed)

        for station, scenario in zip(stations, scenarios):
            strength = station.strength
            if phase == "single_station_drone" and station.station_index == 0:
                strength = 1.0
            audio = generate_synthetic_audio(
                scenario,
                elapsed,
                sample_rate=args.sample_rate,
                window_sec=args.window_sec,
                channels=args.channels,
                station_index=station.station_index,
                strength=strength,
            )
            event = build_event(
                station,
                audio,
                args.sample_rate,
                loop_start,
                metadata_extra={"demo_phase": phase, "demo_scenario": scenario},
            )
            requests.post(args.server, json=event.model_dump(mode="json"), timeout=2.0)
            print(
                f"{phase:32s} {event.station_id} {event.status:11s} "
                f"harm={event.harmonic_score:.1f} agree={event.channel_agreement_count}/{event.channel_count}"
            )

        if not args.realtime and elapsed >= total_duration:
            break
        if args.realtime:
            time.sleep(max(0.0, args.window_sec - (time.time() - loop_start)))
        elif elapsed < total_duration:
            start -= args.window_sec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--realtime", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()
