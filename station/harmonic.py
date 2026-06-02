from __future__ import annotations
import numpy as np
from scipy.signal import get_window

def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))

def harmonic_score(audio, sr, f0_min=500, f0_max=2200, max_freq=7000, min_harmonics=3):
    audio = audio.astype(np.float32)
    audio = audio - np.mean(audio)
    if len(audio) < 1024 or np.max(np.abs(audio)) < 1e-9:
        return 0.0, None, []

    win = get_window("hann", len(audio))
    mag = np.abs(np.fft.rfft(audio * win))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    max_freq = min(max_freq, sr // 2 - 100)
    valid = (freqs >= 100) & (freqs <= max_freq)
    if not np.any(valid):
        return 0.0, None, []

    noise_floor = np.median(mag[valid]) + 1e-12
    best_score, best_f0, best_details = 0.0, None, []

    for f0 in np.arange(f0_min, f0_max + 1, 5):
        vals, weights = [], []
        for k in range(1, 8):
            target = k * f0
            if target > max_freq:
                break
            tol = max(25, min(65, 0.025 * target))
            band = (freqs >= target - tol) & (freqs <= target + tol)
            if np.any(band):
                vals.append(float(np.max(mag[band]) / noise_floor))
                weights.append(1.0 / np.sqrt(k))

        if len(vals) < min_harmonics:
            continue

        vals_arr = np.array(vals)
        weights_arr = np.array(weights)
        balance = min(1.0, (np.median(vals_arr) / (np.max(vals_arr) + 1e-12)) * 4.0)
        score = float(np.sum(db(vals_arr) * weights_arr) / np.sum(weights_arr))
        score *= float(0.65 + 0.35 * balance)

        if score > best_score:
            best_score, best_f0, best_details = score, int(f0), vals

    return best_score, best_f0, best_details
