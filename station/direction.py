from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.signal import butter, sosfiltfilt

SPEED_OF_SOUND = 343.0

def circular_array_positions(n_mics: int, radius_m: float = 0.12) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n_mics, endpoint=False)
    return np.stack([radius_m * np.cos(angles), radius_m * np.sin(angles), np.zeros(n_mics)], axis=1)

def direction_vector(az_deg: float, el_deg: float = 0.0) -> np.ndarray:
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    n = np.arange(len(x))
    return np.interp(n - delay_samples, n, x, left=0, right=0)

def bandpass(x: np.ndarray, sr: int, low: int = 500, high: int = 7000) -> np.ndarray:
    high = min(high, sr // 2 - 100)
    if high <= low:
        return x
    sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x, axis=0)

def delay_and_sum_score(multich: np.ndarray, sr: int, mic_positions: np.ndarray, az_deg: float) -> float:
    d = direction_vector(az_deg)
    delays_sec = mic_positions @ d / SPEED_OF_SOUND
    delays_sec -= delays_sec.min()
    aligned = [fractional_delay(multich[:, ch], -delays_sec[ch] * sr) for ch in range(multich.shape[1])]
    beam = np.mean(np.stack(aligned, axis=1), axis=1)
    return float(np.mean(beam ** 2))

def estimate_azimuth(multich: np.ndarray, sr: int, radius_m: float = 0.12, step_deg: int = 10) -> Tuple[Optional[int], Optional[float]]:
    if multich.ndim != 2 or multich.shape[1] < 3:
        return None, None
    mic_positions = circular_array_positions(multich.shape[1], radius_m)
    try:
        x = bandpass(multich, sr)
    except Exception:
        x = multich
    angles = np.arange(0, 360, step_deg)
    scores = np.array([delay_and_sum_score(x, sr, mic_positions, float(az)) for az in angles])
    best_az = int(angles[np.argmax(scores)])
    confidence = float((scores.max() - np.median(scores)) / (np.std(scores) + 1e-12))
    return best_az, confidence
