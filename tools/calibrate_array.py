from __future__ import annotations

import argparse
import json
from pathlib import Path

from station.array_calibration import build_calibration, write_json
from station.audio_capture import audio_blocks


def parse_int_list(value: str | None) -> list[int] | None:
    if value in (None, ""):
        return None
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate SkyEar array calibration from a field recording.")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--output", default="configs/array_calibration_station_001.json")
    parser.add_argument("--channel-order-report")
    parser.add_argument("--input-channel-order", help="Comma-separated zero-based input order, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--bad-channel", action="append", type=int, default=[])
    parser.add_argument("--min-rms", type=float, default=1e-6)
    parser.add_argument("--dropout-ratio", type=float, default=0.15)
    parser.add_argument("--max-lag-samples", type=int, default=256)
    parser.add_argument("--no-delay", action="store_true")
    return parser.parse_args(argv)


def input_order_from_args(args: argparse.Namespace) -> list[int] | None:
    explicit = parse_int_list(args.input_channel_order)
    if explicit is not None:
        return explicit
    if not args.channel_order_report:
        return None
    payload = json.loads(Path(args.channel_order_report).read_text(encoding="utf-8"))
    order = payload.get("suggested_input_channel_order") or payload.get("detected_input_channel_order")
    return None if order is None else [int(item) for item in order]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Recording {args.duration_sec:.1f}s from {args.channels} channels for calibration...")
    audio = next(
        audio_blocks(
            device_id=args.device_id,
            sample_rate=int(args.sample_rate),
            channels=int(args.channels),
            window_sec=float(args.duration_sec),
        )
    )
    calibration = build_calibration(
        audio,
        sample_rate=int(args.sample_rate),
        input_channel_order=input_order_from_args(args),
        bad_channels=[int(item) for item in args.bad_channel],
        min_rms=float(args.min_rms),
        dropout_ratio=float(args.dropout_ratio),
        estimate_delay=not bool(args.no_delay),
        max_lag_samples=int(args.max_lag_samples),
    )
    write_json(args.output, calibration)
    print(f"wrote {args.output}")
    print("bad_channels:", calibration["bad_channels"])
    print("gain_correction:", ",".join(f"{value:.3f}" for value in calibration["gain_correction"]))
    print("delay_correction_samples:", ",".join(f"{value:.2f}" for value in calibration["delay_correction_samples"]))


if __name__ == "__main__":
    main()
