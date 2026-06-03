from __future__ import annotations

import numpy as np

from station.beamforming import delay_and_sum_beam, estimate_bearing
from station.array_profiles import array_profile
from station.direction import SPEED_OF_SOUND, direction_vector, fractional_delay


def _circular_positions(channels: int = 8, radius_m: float = 0.35) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, channels, endpoint=False)
    return np.stack([radius_m * np.cos(angles), radius_m * np.sin(angles), np.zeros(channels)], axis=1)


def _angle_delta(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _plane_wave(sr: int, bearing_deg: float, freq_hz: float = 700.0, duration_sec: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    positions = _circular_positions()
    t = np.arange(int(sr * duration_sec), dtype=np.float64) / float(sr)
    source = np.sin(2.0 * np.pi * float(freq_hz) * t).astype(np.float32)
    direction = direction_vector(float(bearing_deg))
    delays = positions @ direction / SPEED_OF_SOUND
    audio = np.stack([fractional_delay(source, float(delay) * sr) for delay in delays], axis=1).astype(np.float32)
    return audio, positions


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


def test_delay_and_sum_estimates_plane_wave_60_deg():
    audio, positions = _plane_wave(48000, 60.0)

    result = estimate_bearing(audio, 48000, positions, method="delay_and_sum", scan_step_deg=5, low_hz=500, high_hz=3000)

    assert result.bearing_deg is not None
    assert _angle_delta(result.bearing_deg, 60.0) <= 10.0
    assert result.beam_confidence_pct is not None
    assert 0.0 <= result.beam_confidence_pct <= 1.0


def test_srp_phat_estimates_plane_wave_180_deg():
    audio, positions = _plane_wave(48000, 180.0)

    result = estimate_bearing(audio, 48000, positions, method="srp_phat", scan_step_deg=5, low_hz=500, high_hz=3000)

    assert result.bearing_deg is not None
    assert _angle_delta(result.bearing_deg, 180.0) <= 10.0
    assert result.beam_peak_to_median is not None


def test_random_noise_has_low_confidence_or_high_uncertainty():
    rng = np.random.default_rng(123)
    audio = rng.normal(0.0, 0.02, size=(48000, 8)).astype(np.float32)
    positions = _circular_positions()

    result = estimate_bearing(audio, 48000, positions, scan_step_deg=10, low_hz=500, high_hz=3000)

    assert (result.beam_confidence_pct or 0.0) < 0.45 or (result.bearing_uncertainty_deg or 0.0) > 30.0


def test_mismatched_mic_positions_channel_count_returns_no_bearing():
    audio = np.zeros((4800, 8), dtype=np.float32)
    positions = _circular_positions(channels=7)

    result = estimate_bearing(audio, 48000, positions)

    assert result.bearing_deg is None
    assert result.beam_score is None


def test_delay_and_sum_and_srp_phat_both_run():
    audio, positions = _plane_wave(48000, 90.0)

    delay = estimate_bearing(audio, 48000, positions, method="delay_and_sum", scan_step_deg=15)
    srp = estimate_bearing(audio, 48000, positions, method="srp_phat", scan_step_deg=15)

    assert delay.beamforming_method == "delay_and_sum"
    assert srp.beamforming_method == "srp_phat"
    assert delay.beam_score is not None
    assert srp.beam_score is not None


def test_array_profiles_include_compact_and_field_8ch():
    compact = array_profile("compact_8ch_r0_12m")
    field = array_profile("field_8ch_r0_35m")

    assert compact is not None
    assert field is not None
    assert len(compact["mic_positions_m"]) == 8
    assert len(field["mic_positions_m"]) == 8
    assert field["beamforming"]["high_hz"] == 3000
