from __future__ import annotations

import numpy as np


def compute_spectrum_summary(
    audio_mono: np.ndarray,
    sr: int,
    max_freq: int = 7000,
    n_points: int = 300,
) -> dict[str, list[float] | int]:
    audio = np.asarray(audio_mono, dtype=np.float32).reshape(-1)
    max_freq = int(max_freq)
    if audio.size == 0 or sr <= 0 or n_points <= 0:
        return {
            "spectrum_freqs_hz": [],
            "spectrum_db": [],
            "spectrum_max_freq_hz": max_freq,
        }

    audio = audio - float(np.mean(audio))
    window = np.hanning(audio.size)
    mag = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)

    max_freq = min(max_freq, int(sr // 2))
    keep = freqs <= max_freq
    freqs = freqs[keep]
    mag = mag[keep]

    if freqs.size == 0:
        return {
            "spectrum_freqs_hz": [],
            "spectrum_db": [],
            "spectrum_max_freq_hz": max_freq,
        }

    db = 20 * np.log10(np.maximum(mag, 1e-12))
    db = db - float(np.max(db))

    if freqs.size > n_points:
        idx = np.linspace(0, freqs.size - 1, n_points).astype(int)
        freqs = freqs[idx]
        db = db[idx]

    return {
        "spectrum_freqs_hz": [float(item) for item in freqs],
        "spectrum_db": [float(item) for item in db],
        "spectrum_max_freq_hz": max_freq,
    }


def compute_harmonic_lines(best_f0_hz: int | None, max_freq: int) -> list[dict[str, float | int]]:
    if best_f0_hz is None:
        return []

    f0 = float(best_f0_hz)
    if f0 <= 0:
        return []

    lines = []
    k = 1
    while k * f0 <= max_freq:
        lines.append({"k": k, "freq_hz": float(k * f0)})
        k += 1
    return lines
