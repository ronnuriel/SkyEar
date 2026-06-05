from __future__ import annotations

import numpy as np

from station.audio_filters import HighPassFilter


def test_highpass_filter_reduces_low_frequency_more_than_high_frequency():
    sample_rate = 48000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    low = np.sin(2.0 * np.pi * 60.0 * t)
    high = np.sin(2.0 * np.pi * 1000.0 * t)

    low_filter = HighPassFilter(sample_rate=sample_rate, cutoff_hz=300, channels=1)
    high_filter = HighPassFilter(sample_rate=sample_rate, cutoff_hz=300, channels=1)

    low_out = low_filter.process(low)
    high_out = high_filter.process(high)

    low_rms = float(np.sqrt(np.mean(low_out[int(sample_rate / 2) :] ** 2)))
    high_rms = float(np.sqrt(np.mean(high_out[int(sample_rate / 2) :] ** 2)))

    assert low_rms < 0.05
    assert high_rms > 0.60


def test_highpass_filter_preserves_multichannel_shape():
    audio = np.zeros((2400, 2), dtype=np.float32)
    audio[:, 0] = 0.1
    audio[:, 1] = -0.1
    highpass = HighPassFilter(sample_rate=48000, cutoff_hz=300, channels=2)

    out = highpass.process(audio)

    assert out.shape == audio.shape
