from __future__ import annotations

import pytest

from station.audio_devices import (
    AudioDeviceError,
    generate_station_config,
    resolve_audio_device,
    resolve_channel_count,
    resolve_station_audio_config,
)


DEVICES = [
    {"id": 0, "name": "MacBook Pro Microphone", "max_input_channels": 1, "default_samplerate": 48000},
    {"id": 3, "name": "Universal Audio Volt 2", "max_input_channels": 2, "default_samplerate": 48000},
    {"id": 7, "name": "USB Audio 8ch", "max_input_channels": 8, "default_samplerate": 48000},
]


def test_device_name_resolves_current_device_id():
    device = resolve_audio_device({"device_name": "Volt 2", "device_id": 99}, devices=DEVICES)

    assert device["id"] == 3
    assert device["name"] == "Universal Audio Volt 2"


def test_channels_auto_uses_max_input_channels():
    resolved, device = resolve_station_audio_config(
        {"audio": {"device_match": "USB Audio", "channels": "auto", "sample_rate": "auto"}},
        devices=DEVICES,
    )

    assert device["id"] == 7
    assert resolved["audio"]["channels"] == 8
    assert resolved["audio"]["sample_rate"] == 48000
    assert resolved["audio"]["resolved_max_input_channels"] == 8


def test_device_id_fallback_still_works():
    device = resolve_audio_device({"device_id": 7}, devices=DEVICES)

    assert device["name"] == "USB Audio 8ch"


def test_missing_device_has_helpful_error():
    with pytest.raises(AudioDeviceError) as exc:
        resolve_audio_device({"device_match": "Missing Interface"}, devices=DEVICES)

    assert "Configured audio device was not found" in str(exc.value)
    assert "Input devices:" in str(exc.value)
    assert "Universal Audio Volt 2" in str(exc.value)


def test_channel_count_validation_rejects_too_many_channels():
    with pytest.raises(AudioDeviceError):
        resolve_channel_count({"channels": 9}, DEVICES[2])


def test_volt2_like_device_generates_dual_mic_config():
    cfg = generate_station_config({}, device=DEVICES[1], profile="auto", spacing_m=2.0)

    assert cfg["audio"]["device_id"] == 3
    assert cfg["audio"]["device_name"] == "Universal Audio Volt 2"
    assert cfg["audio"]["channels"] == 2
    assert cfg["audio"]["channel_policy"] == "max_input_channels"
    assert cfg["mic_array"]["profile"] == "volt2_dual_mic"
    assert cfg["mic_array"]["sync_mode"] == "unsynchronized"
    assert cfg["mic_array"]["channel_mode"] == "dual_mic"
    assert cfg["two_mic_direction"]["enabled"] is True
    assert cfg["two_mic_direction"]["left_channel"] == 0
    assert cfg["two_mic_direction"]["right_channel"] == 1
    assert cfg["two_mic_direction"]["spacing_m"] == 2.0


def test_8ch_like_device_generates_circular_clockwise_positions():
    cfg = generate_station_config({}, device=DEVICES[2], profile="auto", radius_m=0.35, channel_0_heading_deg=0.0)

    positions = cfg["mic_array"]["mic_positions_m"]
    assert cfg["audio"]["channels"] == 8
    assert cfg["mic_array"]["profile"] == "array_8ch_clockwise_north"
    assert cfg["mic_array"]["sync_mode"] == "synchronized"
    assert cfg["mic_array"]["channel_order"] == "clockwise"
    assert cfg["mic_array"]["channel_0_heading_deg"] == 0.0
    assert cfg["mic_array"]["radius_m"] == 0.35
    assert len(positions) == 8
    assert positions[0] == [0.35, 0.0, 0.0]
    assert cfg["beamforming"]["enabled"] is True
    assert cfg["two_mic_direction"]["enabled"] is False


def test_mono_device_generates_mono_profile():
    cfg = generate_station_config({}, device=DEVICES[0], profile="auto")

    assert cfg["audio"]["channels"] == 1
    assert cfg["mic_array"]["profile"] == "mono"
    assert cfg["mic_array"]["sync_mode"] == "mono"
    assert cfg["two_mic_direction"]["enabled"] is False
