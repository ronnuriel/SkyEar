from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

from station.audio_capture import CapturedAudioBlock, audio_block_stream, list_input_devices, per_channel_rms


class AudioDeviceError(RuntimeError):
    pass


def render_input_devices(devices: list[dict[str, Any]]) -> str:
    lines = ["Input devices:"]
    for device in devices:
        default = " default" if device.get("is_default_input") else ""
        hostapi = f" hostapi={device.get('hostapi')}" if device.get("hostapi") else ""
        lines.append(
            f"[{device.get('id')}] {device.get('name')} "
            f"inputs={device.get('max_input_channels')} sr={device.get('default_samplerate')}{hostapi}{default}"
        )
    return "\n".join(lines)


def resolve_audio_device(audio_cfg: dict[str, Any], devices: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    devices = list_input_devices() if devices is None else list(devices)
    if not devices:
        raise AudioDeviceError("No input audio devices found.")

    match = audio_cfg.get("device_match") or audio_cfg.get("device_name")
    if match:
        needle = str(match).strip().lower()
        for device in devices:
            if needle and needle in str(device.get("name") or "").lower():
                return dict(device)

    device_id = audio_cfg.get("device_id")
    if device_id not in (None, ""):
        try:
            wanted = int(device_id)
        except (TypeError, ValueError):
            wanted = None
        if wanted is not None:
            for device in devices:
                if int(device.get("id")) == wanted:
                    return dict(device)

    detail = render_input_devices(devices)
    raise AudioDeviceError(
        "Configured audio device was not found. Set audio.device_name/device_match or audio.device_id.\n" + detail
    )


def resolve_channel_count(audio_cfg: dict[str, Any], device: dict[str, Any]) -> int:
    max_channels = int(device.get("max_input_channels") or 0)
    configured = audio_cfg.get("channels")
    if configured in (None, "", "auto"):
        if max_channels <= 0:
            raise AudioDeviceError(f"Device {device.get('name')} has no input channels.")
        return max_channels
    channels = int(configured)
    if channels <= 0:
        raise AudioDeviceError("audio.channels must be positive or 'auto'.")
    if max_channels > 0 and channels > max_channels:
        raise AudioDeviceError(
            f"Configured audio.channels={channels} exceeds device max_input_channels={max_channels} for {device.get('name')}."
        )
    return channels


def resolve_station_audio_config(
    cfg: dict[str, Any],
    devices: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = copy.deepcopy(cfg)
    audio_cfg = resolved.setdefault("audio", {})
    device = resolve_audio_device(audio_cfg, devices=devices)
    channels = resolve_channel_count(audio_cfg, device)
    sample_rate = audio_cfg.get("sample_rate")
    if sample_rate in (None, "", "auto"):
        sample_rate = int(device.get("default_samplerate") or 48000)
    audio_cfg.update(
        {
            "device_id": int(device["id"]),
            "device_name": str(device.get("name") or ""),
            "sample_rate": int(float(sample_rate)),
            "channels": int(channels),
            "resolved_max_input_channels": int(device.get("max_input_channels") or channels),
        }
    )
    return resolved, device


def infer_profile(device: dict[str, Any], channels: int, requested: str = "auto") -> str:
    requested = str(requested or "auto")
    if requested != "auto":
        return requested
    name = str(device.get("name") or "").lower()
    if channels <= 1:
        return "mono"
    if channels == 2:
        return "volt2_dual_mic" if "volt" in name else "generic_dual_mic"
    if channels == 8:
        return "array_8ch_clockwise_north"
    return "circular_clockwise"


def circular_clockwise_positions(
    channels: int,
    radius_m: float,
    channel_0_heading_deg: float = 0.0,
) -> list[list[float]]:
    positions: list[list[float]] = []
    for idx in range(int(channels)):
        angle = math.radians(float(channel_0_heading_deg) - idx * 360.0 / float(channels))
        positions.append(
            [
                float(radius_m * math.cos(angle)),
                float(radius_m * math.sin(angle)),
                0.0,
            ]
        )
    return positions


def generate_station_config(
    base_cfg: dict[str, Any],
    *,
    device: dict[str, Any],
    profile: str = "auto",
    channels: int | str | None = None,
    sample_rate: int | None = None,
    spacing_m: float = 2.0,
    front_heading_deg: float | None = None,
    radius_m: float = 0.35,
    channel_0_heading_deg: float = 0.0,
    station_id: str | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    selected_channels = int(device.get("max_input_channels") if channels in (None, "auto") else channels)
    selected_sample_rate = int(sample_rate or device.get("default_samplerate") or 48000)
    profile_name = infer_profile(device, selected_channels, requested=profile)
    cfg.setdefault("station", {})
    if station_id:
        cfg["station"]["station_id"] = station_id
        cfg["station"]["id"] = station_id
    cfg["audio"] = {
        **(cfg.get("audio") or {}),
        "device_id": int(device["id"]),
        "device_name": str(device.get("name") or ""),
        "device_match": str(device.get("name") or ""),
        "sample_rate": selected_sample_rate,
        "channels": selected_channels,
        "channel_policy": "max_input_channels" if channels in (None, "auto") else "manual",
        "latency": "high",
        "capture_block_sec": 0.25,
    }
    mic_cfg = cfg.setdefault("mic_array", {})
    if selected_channels <= 1 or profile_name == "mono":
        mic_cfg.update({"profile": "mono", "sync_mode": "mono", "channel_mode": "mono"})
        cfg.setdefault("two_mic_direction", {})["enabled"] = False
        return cfg

    if profile_name in {"volt2_dual_mic", "generic_dual_mic"}:
        mic_cfg.update({"profile": profile_name, "sync_mode": "unsynchronized", "channel_mode": "dual_mic"})
        two_mic = cfg.setdefault("two_mic_direction", {})
        two_mic.update(
            {
                "enabled": True,
                "left_channel": 0,
                "right_channel": 1,
                "spacing_m": float(spacing_m),
                "center_deadzone_deg": 12,
                "look_sector_width_deg": 60,
                "front_heading_deg": front_heading_deg,
            }
        )
        return cfg

    profile_for_array = "array_8ch_clockwise_north" if profile_name == "array_8ch_clockwise_north" else "circular_clockwise"
    mic_cfg.update(
        {
            "profile": profile_for_array,
            "channels": selected_channels,
            "sync_mode": "synchronized",
            "channel_mode": "array",
            "channel_order": "clockwise",
            "channel_0_heading_deg": float(channel_0_heading_deg),
            "heading_reference": "geographic_north",
            "radius_m": float(radius_m),
            "mic_positions_m": circular_clockwise_positions(selected_channels, float(radius_m), float(channel_0_heading_deg)),
        }
    )
    cfg.setdefault("beamforming", {})["enabled"] = True
    cfg.setdefault("two_mic_direction", {})["enabled"] = False
    return cfg


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def sample_audio_health(
    *,
    device_id: int | None,
    sample_rate: int,
    channels: int,
    seconds: float = 5.0,
    capture_block_sec: float = 0.25,
    latency: str | float | None = "high",
) -> dict[str, Any]:
    overflow_count = 0
    rms_accum: list[list[float]] = []
    total_sec = 0.0
    for block in audio_block_stream(
        device_id=device_id,
        sample_rate=sample_rate,
        channels=channels,
        window_sec=capture_block_sec,
        latency=latency,
    ):
        overflow_count += 1 if block.input_overflow else 0
        rms_accum.append(per_channel_rms(block.audio))
        total_sec += max(0.0, float(block.end_unix) - float(block.start_unix))
        if total_sec >= float(seconds):
            break
    if not rms_accum:
        rms = []
    else:
        width = max(len(row) for row in rms_accum)
        rms = []
        for idx in range(width):
            values = [row[idx] for row in rms_accum if idx < len(row)]
            rms.append(float(sum(values) / max(1, len(values))))
    return {"overflow_count": overflow_count, "rms_per_channel": rms, "duration_sec": total_sec}
