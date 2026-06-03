from __future__ import annotations

import numpy as np

from station.beamforming import delay_and_sum_beam, estimate_bearing


def test_delay_and_sum_returns_mono_beam():
    sr = 48000
    audio = np.zeros((sr // 10, 2), dtype=np.float32)
    audio[:, 0] = 0.1
    audio[:, 1] = 0.1
    positions = np.asarray([[0.0, 0.0, 0.0], [0.12, 0.0, 0.0]])

    beam = delay_and_sum_beam(audio, sr, positions, 0.0)

    assert beam.shape == (sr // 10,)


def test_estimate_bearing_returns_field_ready_metrics():
    sr = 48000
    t = np.arange(sr // 5) / sr
    tone = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
    audio = np.stack([tone, np.roll(tone, 4), np.roll(tone, 8), np.roll(tone, 12)], axis=1)
    positions = np.asarray(
        [
            [0.10, 0.0, 0.0],
            [0.0, 0.10, 0.0],
            [-0.10, 0.0, 0.0],
            [0.0, -0.10, 0.0],
        ]
    )

    result = estimate_bearing(audio, sr, positions, scan_step_deg=30)

    assert result.beamforming_method == "delay_and_sum"
    assert result.bearing_deg is None or 0.0 <= result.bearing_deg < 360.0
    assert result.beam_score is None or result.beam_score >= 0.0
