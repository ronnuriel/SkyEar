from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from station.audio_capture import list_input_devices
from station.audio_devices import AudioDeviceError, render_input_devices, resolve_audio_device, resolve_channel_count, sample_audio_health


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SkyEar audio device configuration and capture health.")
    parser.add_argument("--config", default="configs/config_station.yaml")
    parser.add_argument("--seconds", "--diagnostic-sec", dest="seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _load_config(args.config)
    audio_cfg = cfg.get("audio", {}) or {}
    devices = list_input_devices()
    print(render_input_devices(devices))
    try:
        device = resolve_audio_device(audio_cfg, devices=devices)
        channels = resolve_channel_count(audio_cfg, device)
    except AudioDeviceError as exc:
        print(f"\nFAILED: {exc}")
        return 2
    sample_rate = int(audio_cfg.get("sample_rate") or device.get("default_samplerate") or 48000)
    capture_block_sec = float(audio_cfg.get("capture_block_sec") or audio_cfg.get("window_sec") or 0.25)
    latency = audio_cfg.get("latency", "high")
    print("\nConfigured audio:")
    print(f"device: {device.get('name')}")
    print(f"device_id: {device.get('id')}")
    print(f"expected_channels: {channels}")
    print(f"actual_max_input_channels: {device.get('max_input_channels')}")
    print(f"sample_rate: {sample_rate}")
    print(f"latency: {latency}")
    print(f"capture_block_sec: {capture_block_sec}")
    if args.dry_run:
        return 0
    health = sample_audio_health(
        device_id=int(device["id"]),
        sample_rate=sample_rate,
        channels=channels,
        seconds=float(args.seconds),
        capture_block_sec=capture_block_sec,
        latency=latency,
    )
    print("\nCapture health:")
    print(f"duration_sec: {health['duration_sec']:.2f}")
    print(f"overflow_count: {health['overflow_count']}")
    print(f"rms_per_channel: {health['rms_per_channel']}")
    return 0


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    raise SystemExit(main())
