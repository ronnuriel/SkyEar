from __future__ import annotations

import numpy as np

from station.two_mic_direction import estimate_two_mic_side


def _delayed_pair(delay_samples: int, sample_rate: int = 48000) -> np.ndarray:
    rng = np.random.default_rng(42)
    source = rng.normal(0.0, 0.2, sample_rate // 2).astype(np.float32)
    left = np.zeros_like(source)
    right = np.zeros_like(source)
    if delay_samples >= 0:
        left[:] = source
        right[delay_samples:] = source[: source.size - delay_samples]
    else:
        delay = abs(delay_samples)
        right[:] = source
        left[delay:] = source[: source.size - delay]
    return np.stack([left, right], axis=1)


def test_left_side_when_left_channel_arrives_first():
    result = estimate_two_mic_side(
        _delayed_pair(12),
        48000,
        spacing_m=0.5,
        min_delay_us=40,
        min_peak_ratio=1.1,
    )

    assert result.side == "left"
    assert result.delay_us is not None
    assert result.delay_us > 0
    assert result.confidence and result.confidence > 0.35


def test_right_side_when_right_channel_arrives_first():
    result = estimate_two_mic_side(
        _delayed_pair(-12),
        48000,
        spacing_m=0.5,
        min_delay_us=40,
        min_peak_ratio=1.1,
    )

    assert result.side == "right"
    assert result.delay_us is not None
    assert result.delay_us < 0


def test_center_when_delay_is_small():
    result = estimate_two_mic_side(
        _delayed_pair(0),
        48000,
        spacing_m=0.5,
        min_delay_us=40,
        min_peak_ratio=1.1,
    )

    assert result.side == "center"
    assert abs(float(result.delay_us or 0.0)) < 40.0


def test_uncertain_when_signal_is_too_low():
    result = estimate_two_mic_side(
        np.zeros((4800, 2), dtype=np.float32),
        48000,
        spacing_m=0.5,
    )

    assert result.side == "uncertain"
    assert result.reason == "low_signal"
