from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from station.audio_capture import audio_block_stream
from station.two_mic_direction import TwoMicDirectionResult, estimate_two_mic_side
from station.two_mic_direction_tracker import TwoMicDirectionTracker, two_mic_tracker_config_from_dict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live practical left/right/center check for a two-mic station.")
    parser.add_argument("--config", default="configs/config_station.yaml")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--count", type=int, default=0, help="Number of windows to print; 0 means run forever.")
    parser.add_argument("--tracked", action="store_true", help="Apply the same short stability tracker used by the station.")
    parser.add_argument("--dry-run", action="store_true", help="Print instructions and resolved settings, then exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _load_config(args.config)
    audio_cfg = cfg.get("audio", {}) or {}
    two_mic_cfg = cfg.get("two_mic_direction", {}) or {}
    sample_rate = int(args.sample_rate or audio_cfg.get("sample_rate", 48000))
    channels = int(args.channels or audio_cfg.get("channels", 2))
    device_id = args.device_id if args.device_id is not None else audio_cfg.get("device_id")
    window_sec = float(args.window_sec or audio_cfg.get("window_sec", 1.0))

    _print_instructions(two_mic_cfg, sample_rate, channels, device_id, window_sec, tracked=bool(args.tracked))
    if args.dry_run:
        return 0
    if channels < 2:
        raise SystemExit("two-mic direction check requires at least 2 input channels")

    tracker = TwoMicDirectionTracker(two_mic_tracker_config_from_dict(two_mic_cfg)) if args.tracked else None
    stream = audio_block_stream(device_id, sample_rate, channels, window_sec)
    printed = 0
    for block in stream:
        raw = _estimate(block.audio, sample_rate, two_mic_cfg)
        result = tracker.update(block.end_unix, raw) if tracker is not None else raw
        print(_format_result(result))
        printed += 1
        if int(args.count) > 0 and printed >= int(args.count):
            break
    return 0


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload or {}


def _estimate(audio, sample_rate: int, cfg: dict[str, Any]) -> TwoMicDirectionResult:
    return estimate_two_mic_side(
        audio,
        sample_rate,
        spacing_m=float(cfg.get("spacing_m", 2.0)),
        left_channel=int(cfg.get("left_channel", 0)),
        right_channel=int(cfg.get("right_channel", 1)),
        low_hz=int(cfg.get("low_hz", 700)),
        high_hz=int(cfg.get("high_hz", 6000)),
        min_delay_us=float(cfg.get("min_delay_us", 40.0)),
        center_deadzone_deg=_optional_float(cfg.get("center_deadzone_deg")),
        look_sector_width_deg=float(cfg.get("look_sector_width_deg", 60.0)),
        unstable_sector_width_deg=float(cfg.get("unstable_sector_width_deg", 120.0)),
        far_side_angle_deg=float(cfg.get("far_side_angle_deg", 55.0)),
        min_peak_ratio=float(cfg.get("min_peak_ratio", 1.4)),
        min_rms=float(cfg.get("min_rms", 0.0005)),
        front_heading_deg=_optional_float(cfg.get("front_heading_deg")),
    )


def _print_instructions(
    cfg: dict[str, Any],
    sample_rate: int,
    channels: int,
    device_id: int | None,
    window_sec: float,
    *,
    tracked: bool,
) -> None:
    spacing_m = float(cfg.get("spacing_m", 2.0))
    left_channel = int(cfg.get("left_channel", 0))
    right_channel = int(cfg.get("right_channel", 1))
    center_deadzone = _optional_float(cfg.get("center_deadzone_deg"))
    print("Two-mic direction check")
    print("Instructions:")
    print("- Clap near the left mic -> should show LOOK LEFT")
    print("- Clap near the right mic -> should show LOOK RIGHT")
    print("- Clap centered in front -> should show LOOK CENTER")
    print("- Two mics are front/back ambiguous; this is a search hint, not a 360 bearing")
    print(
        "settings "
        f"device_id={device_id} sample_rate={sample_rate} channels={channels} "
        f"window_sec={window_sec:g} spacing_m={spacing_m:g} "
        f"left_channel={left_channel} right_channel={right_channel} "
        f"center_deadzone_deg={_fmt(center_deadzone)} tracked={int(tracked)}"
    )


def _format_result(result: TwoMicDirectionResult) -> str:
    parts = [
        f"time={time.strftime('%H:%M:%S')}",
        f"side={result.side}",
        f"delay_us={_fmt(result.delay_us, digits=0)}",
        f"angle_from_center_deg={_fmt(result.angle_from_center_deg, digits=1)}",
        f"confidence={_fmt(result.confidence, digits=2)}",
        f"peak_ratio={_fmt(result.peak_ratio, digits=2)}",
        f"stable={int(bool(result.stable))}",
        f"front_back_ambiguous={int(bool(result.front_back_ambiguous))}",
        f"look={json.dumps(result.look_hint or 'DIRECTION UNKNOWN - scan left and right, front/back ambiguous')}",
    ]
    if result.possible_front_azimuth_deg is not None and result.possible_back_azimuth_deg is not None:
        parts.append(f"possible_front_azimuth_deg={_fmt(result.possible_front_azimuth_deg, digits=1)}")
        parts.append(f"possible_back_azimuth_deg={_fmt(result.possible_back_azimuth_deg, digits=1)}")
    if result.reason:
        parts.append(f"reason={result.reason}")
    return " ".join(parts)


def _fmt(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return float(text)


if __name__ == "__main__":
    raise SystemExit(main())
