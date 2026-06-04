from __future__ import annotations

import argparse
import time
from pathlib import Path

from station.array_calibration import detect_strongest_channel, verify_channel_order, write_json
from station.audio_capture import audio_blocks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check physical 8-channel array channel order by tap/clap.")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--tap-sec", type=float, default=1.5)
    parser.add_argument("--output", default="channel_order_report.json")
    parser.add_argument("--non-interactive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    detected: list[int] = []
    stream = audio_blocks(
        device_id=args.device_id,
        sample_rate=int(args.sample_rate),
        channels=int(args.channels),
        window_sec=float(args.tap_sec),
    )
    for mic_idx in range(int(args.channels)):
        if not args.non_interactive:
            input(f"Tap/clap near physical mic {mic_idx + 1}, then press Enter...")
        else:
            print(f"Listening for physical mic {mic_idx + 1} tap/clap...")
        time.sleep(0.05)
        audio = next(stream)
        strongest = detect_strongest_channel(audio)
        detected.append(strongest)
        print(f"physical_mic={mic_idx + 1} strongest_input_channel={strongest + 1}")

    report = verify_channel_order(detected, int(args.channels))
    report.update(
        {
            "sample_rate": int(args.sample_rate),
            "tap_sec": float(args.tap_sec),
            "output_note": "input channels are zero-based in JSON; printed channels are one-based",
        }
    )
    write_json(Path(args.output), report)
    print(f"wrote {args.output}")
    if not report["correct"]:
        print("Channel order mismatch detected.")
        print("Use suggested_input_channel_order as input_channel_order in calibration JSON.")


if __name__ == "__main__":
    main()
