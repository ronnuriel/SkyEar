from __future__ import annotations

import numpy as np

from station.spectrum import compute_harmonic_lines, compute_spectrum_summary


def test_spectrum_summary_returns_equal_length_arrays():
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    audio = np.sin(2 * np.pi * 1000 * t)

    summary = compute_spectrum_summary(audio, sr, n_points=300)

    assert len(summary["spectrum_freqs_hz"]) == len(summary["spectrum_db"])
    assert summary["spectrum_max_freq_hz"] == 7000


def test_spectrum_summary_downsamples_to_n_points_or_less():
    sr = 44100
    audio = np.zeros(sr, dtype=np.float32)

    summary = compute_spectrum_summary(audio, sr, n_points=120)

    assert len(summary["spectrum_freqs_hz"]) <= 120
    assert len(summary["spectrum_db"]) <= 120


def test_harmonic_lines_stop_at_max_freq():
    lines = compute_harmonic_lines(1000, 5000)

    assert [line["freq_hz"] for line in lines] == [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]


def test_harmonic_lines_none_returns_empty_list():
    assert compute_harmonic_lines(None, 5000) == []
