from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from station.audio_capture import list_input_devices
from station.audio_devices import generate_station_config, render_input_devices, write_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure SkyEar audio device and microphone profile.")
    parser.add_argument("--config", default="configs/config_station.yaml")
    parser.add_argument("--output")
    parser.add_argument("--profile", default="auto", choices=["auto", "mono", "volt2_dual_mic", "generic_dual_mic", "circular_clockwise", "array_8ch_clockwise_north"])
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--channels")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--spacing-m", type=float, default=2.0)
    parser.add_argument("--front-heading-deg", type=float)
    parser.add_argument("--radius-m", type=float, default=0.35)
    parser.add_argument("--channel-0-heading-deg", type=float, default=0.0)
    parser.add_argument("--station-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    devices = list_input_devices()
    if not devices:
        print("No input audio devices found.")
        return 2
    print(render_input_devices(devices))
    device = _select_device(devices, args)
    channels = _select_channels(device, args)
    sample_rate = _select_sample_rate(device, args)
    base = _load_yaml(args.config)
    generated = generate_station_config(
        base,
        device=device,
        profile=str(args.profile),
        channels=channels,
        sample_rate=sample_rate,
        spacing_m=float(args.spacing_m),
        front_heading_deg=args.front_heading_deg,
        radius_m=float(args.radius_m),
        channel_0_heading_deg=float(args.channel_0_heading_deg),
        station_id=args.station_id,
    )
    output_path = args.output or args.config
    text = yaml.safe_dump(generated, sort_keys=False)
    print("\nGenerated config preview:")
    print(text)
    if args.dry_run:
        return 0
    if not args.non_interactive and not _confirm(f"Save config to {output_path}? [Y/n]: ", default=True):
        print("Not saved.")
        return 0
    write_yaml(output_path, generated)
    print(f"Saved {output_path}")
    return 0


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _select_device(devices: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if args.device_id is not None:
        for device in devices:
            if int(device["id"]) == int(args.device_id):
                return dict(device)
        raise SystemExit(f"device_id={args.device_id} was not found")
    if args.non_interactive or args.dry_run:
        return dict(next((device for device in devices if device.get("is_default_input")), devices[0]))
    while True:
        raw = input("Choose input device: ").strip()
        try:
            wanted = int(raw)
        except ValueError:
            print("Please enter a device index.")
            continue
        for device in devices:
            if int(device["id"]) == wanted:
                return dict(device)
        print(f"No input device with index {wanted}.")


def _select_channels(device: dict[str, Any], args: argparse.Namespace) -> int:
    max_inputs = int(device.get("max_input_channels") or 1)
    if args.channels not in (None, ""):
        return max(1, min(max_inputs, int(args.channels)))
    if args.non_interactive or args.dry_run:
        return max_inputs
    if _confirm("Use max input channels? [Y/n]: ", default=True):
        return max_inputs
    while True:
        raw = input(f"Channels [1-{max_inputs}]: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a channel count.")
            continue
        if 1 <= value <= max_inputs:
            return value
        print(f"Choose between 1 and {max_inputs}.")


def _select_sample_rate(device: dict[str, Any], args: argparse.Namespace) -> int:
    default_sr = int(device.get("default_samplerate") or 48000)
    if args.sample_rate:
        return int(args.sample_rate)
    if args.non_interactive or args.dry_run:
        return default_sr
    raw = input(f"Sample rate [{default_sr}]: ").strip()
    return default_sr if not raw else int(raw)


def _confirm(prompt: str, *, default: bool) -> bool:
    raw = input(prompt).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"y", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
