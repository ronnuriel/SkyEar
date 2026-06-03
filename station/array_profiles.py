from __future__ import annotations

import numpy as np


def circular_mic_positions(channels: int, radius_m: float) -> list[list[float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, int(channels), endpoint=False)
    return [
        [float(radius_m * np.cos(angle)), float(radius_m * np.sin(angle)), 0.0]
        for angle in angles
    ]


ARRAY_PROFILES: dict[str, dict] = {
    "mac_builtin_mono": {
        "channels": 1,
        "sync_mode": "mono",
        "channel_mode": "mono",
    },
    "remote_mono": {
        "channels": 1,
        "sync_mode": "mono",
        "channel_mode": "mono",
    },
    "compact_8ch_r0_12m": {
        "channels": 8,
        "radius_m": 0.12,
        "sync_mode": "synchronized",
        "mic_positions_m": circular_mic_positions(8, 0.12),
        "beamforming": {"low_hz": 500, "high_hz": 3000},
    },
    "field_8ch_r0_35m": {
        "channels": 8,
        "radius_m": 0.35,
        "sync_mode": "synchronized",
        "mic_positions_m": circular_mic_positions(8, 0.35),
        "beamforming": {"low_hz": 500, "high_hz": 3000},
    },
}


def array_profile(name: str) -> dict | None:
    profile = ARRAY_PROFILES.get(name)
    return None if profile is None else dict(profile)
