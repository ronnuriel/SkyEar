from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from station.direction import SPEED_OF_SOUND, bandpass


@dataclass
class TwoMicDirectionResult:
    side: str = "unavailable"
    delay_us: float | None = None
    confidence: float | None = None
    peak_ratio: float | None = None
    reason: str | None = None


def estimate_two_mic_side(
    audio: np.ndarray,
    sample_rate: int,
    *,
    spacing_m: float,
    left_channel: int = 0,
    right_channel: int = 1,
    low_hz: int = 500,
    high_hz: int = 6000,
    min_delay_us: float = 40.0,
    min_peak_ratio: float = 1.2,
    min_rms: float = 1e-5,
) -> TwoMicDirectionResult:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        return TwoMicDirectionResult(reason="requires_two_channels")
    if sample_rate <= 0:
        return TwoMicDirectionResult(reason="invalid_sample_rate")
    if spacing_m <= 0.0:
        return TwoMicDirectionResult(reason="invalid_spacing")
    if left_channel == right_channel or max(left_channel, right_channel) >= array.shape[1]:
        return TwoMicDirectionResult(reason="invalid_channels")

    left = array[:, int(left_channel)]
    right = array[:, int(right_channel)]
    rms = max(_rms(left), _rms(right))
    if rms < float(min_rms):
        return TwoMicDirectionResult(side="uncertain", reason="low_signal")

    pair = np.stack([left, right], axis=1)
    try:
        pair = bandpass(pair, int(sample_rate), low=int(low_hz), high=int(high_hz))
    except Exception:
        pass
    left = pair[:, 0] - float(np.mean(pair[:, 0]))
    right = pair[:, 1] - float(np.mean(pair[:, 1]))

    max_delay_sec = float(spacing_m) / SPEED_OF_SOUND
    max_lag = max(1, int(np.ceil(max_delay_sec * int(sample_rate))))
    corr = correlate(right, left, mode="full", method="fft")
    lags = np.arange(-left.size + 1, right.size)
    keep = np.abs(lags) <= max_lag
    corr = np.abs(corr[keep])
    lags = lags[keep]
    if corr.size == 0 or not np.isfinite(corr).any():
        return TwoMicDirectionResult(side="uncertain", reason="no_correlation")

    peak_idx = int(np.nanargmax(corr))
    lag = int(lags[peak_idx])
    delay_us = float(lag / int(sample_rate) * 1_000_000.0)
    peak = float(corr[peak_idx])
    median = float(np.nanmedian(corr) + 1e-12)
    peak_ratio = float(peak / median)
    if peak_ratio < float(min_peak_ratio):
        return TwoMicDirectionResult(
            side="uncertain",
            delay_us=delay_us,
            confidence=0.0,
            peak_ratio=peak_ratio,
            reason="weak_correlation_peak",
        )

    if abs(delay_us) < float(min_delay_us):
        side = "center"
    elif delay_us > 0.0:
        side = "left"
    else:
        side = "right"

    delay_confidence = min(1.0, abs(delay_us) / max(float(min_delay_us), max_delay_sec * 1_000_000.0))
    peak_confidence = min(1.0, max(0.0, peak_ratio - float(min_peak_ratio)) / 3.0)
    confidence = float(np.clip(0.35 + 0.35 * delay_confidence + 0.30 * peak_confidence, 0.0, 1.0))
    return TwoMicDirectionResult(
        side=side,
        delay_us=delay_us,
        confidence=confidence,
        peak_ratio=peak_ratio,
        reason=None,
    )


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
