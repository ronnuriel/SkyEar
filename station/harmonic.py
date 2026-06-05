from __future__ import annotations
import numpy as np
from scipy.signal import get_window

def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))

def harmonic_score(
    audio,
    sr,
    f0_min=500,
    f0_max=2200,
    max_freq=7000,
    min_harmonics=3,
    min_ridge_prominence_db=6.0,
):
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
        vals, weights, details = [], [], []
        for k in range(1, 8):
            target = k * f0
            if target > max_freq:
                break
            tol = max(25, min(65, 0.025 * target))
            band = (freqs >= target - tol) & (freqs <= target + tol)
            if np.any(band):
                band_idx = np.where(band)[0]
                peak_local_idx = int(np.argmax(mag[band]))
                peak_idx = int(band_idx[peak_local_idx])
                peak_mag = float(mag[peak_idx])
                side_low = (freqs >= target - 4.0 * tol) & (freqs < target - 1.4 * tol)
                side_high = (freqs > target + 1.4 * tol) & (freqs <= target + 4.0 * tol)
                side = side_low | side_high
                local_floor = float(np.median(mag[side])) if np.any(side) else float(noise_floor)
                local_floor = max(local_floor, float(noise_floor), 1e-12)
                prominence_db = float(db(peak_mag / local_floor))
                present = prominence_db >= float(min_ridge_prominence_db)
                ratio = float(peak_mag / noise_floor)
                vals.append(ratio)
                weights.append(1.0 / np.sqrt(k))
                details.append(
                    {
                        "k": int(k),
                        "target_hz": float(target),
                        "peak_hz": float(freqs[peak_idx]),
                        "ratio": ratio,
                        "prominence_db": prominence_db,
                        "present": bool(present),
                    }
                )

        present_count = sum(1 for item in details if item["present"])
        if present_count < min_harmonics:
            continue

        vals_arr = np.array(vals)
        weights_arr = np.array(weights)
        present_weights = np.array([weight if item["present"] else weight * 0.25 for weight, item in zip(weights, details)])
        balance = min(1.0, (np.median(vals_arr) / (np.max(vals_arr) + 1e-12)) * 4.0)
        score = float(np.sum(db(vals_arr) * present_weights) / np.sum(weights_arr))
        score *= float(0.65 + 0.35 * balance)
        score *= float(min(1.0, present_count / max(1, int(min_harmonics))))

        if score > best_score:
            best_score, best_f0, best_details = score, int(f0), details

    return best_score, best_f0, best_details
