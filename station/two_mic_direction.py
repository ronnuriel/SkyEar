from __future__ import annotations

from dataclasses import dataclass
import math

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
    angle_from_center_deg: float | None = None
    look_label: str = "unknown"
    look_hint: str | None = None
    sector_center_deg: float | None = None
    sector_width_deg: float | None = None
    front_back_ambiguous: bool = True
    stable: bool = False
    possible_front_azimuth_deg: float | None = None
    possible_back_azimuth_deg: float | None = None
    stable_window_count: int = 0
    tracker_window_count: int = 0


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
    center_deadzone_deg: float | None = None,
    look_sector_width_deg: float = 60.0,
    unstable_sector_width_deg: float = 120.0,
    far_side_angle_deg: float = 55.0,
    min_peak_ratio: float = 1.2,
    min_rms: float = 1e-5,
    front_heading_deg: float | None = None,
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
        return _unknown_result("low_signal", unstable_sector_width_deg)

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
        return _unknown_result("no_correlation", unstable_sector_width_deg)

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
            look_label="unknown",
            look_hint="DIRECTION UNKNOWN - scan left and right, front/back ambiguous",
            sector_width_deg=float(unstable_sector_width_deg),
        )

    angle = _angle_from_delay(delay_us, spacing_m)
    if angle is None:
        return _unknown_result("invalid_delay", unstable_sector_width_deg)
    if center_deadzone_deg is not None:
        center_delay_us = _center_delay_us(float(spacing_m), float(center_deadzone_deg))
    else:
        center_delay_us = float(min_delay_us)
        center_deadzone_deg = abs(float(angle)) if abs(float(delay_us)) <= center_delay_us else 0.0

    if abs(delay_us) <= float(center_delay_us):
        side = "center"
    elif delay_us > 0.0:
        side = "left"
    else:
        side = "right"

    delay_confidence = min(1.0, abs(delay_us) / max(float(center_delay_us), max_delay_sec * 1_000_000.0))
    peak_confidence = min(1.0, max(0.0, peak_ratio - float(min_peak_ratio)) / 3.0)
    confidence = float(np.clip(0.35 + 0.35 * delay_confidence + 0.30 * peak_confidence, 0.0, 1.0))
    look_label, look_hint, sector_center, sector_width = _look_hint(
        side=side,
        angle_from_center_deg=angle,
        center_deadzone_deg=float(center_deadzone_deg),
        look_sector_width_deg=float(look_sector_width_deg),
        far_side_angle_deg=float(far_side_angle_deg),
    )
    possible_front, possible_back = _possible_azimuths(front_heading_deg, angle)
    return TwoMicDirectionResult(
        side=side,
        delay_us=delay_us,
        confidence=confidence,
        peak_ratio=peak_ratio,
        reason=None,
        angle_from_center_deg=angle,
        look_label=look_label,
        look_hint=look_hint,
        sector_center_deg=sector_center,
        sector_width_deg=sector_width,
        front_back_ambiguous=True,
        stable=True,
        possible_front_azimuth_deg=possible_front,
        possible_back_azimuth_deg=possible_back,
    )


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))


def _angle_from_delay(delay_us: float, spacing_m: float) -> float | None:
    max_delay_sec = float(spacing_m) / SPEED_OF_SOUND
    if max_delay_sec <= 0.0:
        return None
    delay_sec = float(delay_us) / 1_000_000.0
    x = float(np.clip(delay_sec / max_delay_sec, -1.0, 1.0))
    return float(math.degrees(math.asin(x)))


def _center_delay_us(spacing_m: float, center_deadzone_deg: float) -> float:
    max_delay_sec = float(spacing_m) / SPEED_OF_SOUND
    deadzone_rad = math.radians(max(0.0, min(89.0, float(center_deadzone_deg))))
    return float(max_delay_sec * math.sin(deadzone_rad) * 1_000_000.0)


def _look_hint(
    *,
    side: str,
    angle_from_center_deg: float,
    center_deadzone_deg: float,
    look_sector_width_deg: float,
    far_side_angle_deg: float,
) -> tuple[str, str, float | None, float]:
    angle = float(angle_from_center_deg)
    abs_angle = abs(angle)
    sector = float(look_sector_width_deg)
    scan_half = max(1.0, sector / 2.0)
    if abs_angle <= float(center_deadzone_deg) or side == "center":
        return (
            "center",
            f"LOOK CENTER - scan +/-{scan_half:.0f} deg, front/back ambiguous",
            0.0,
            sector,
        )
    if angle > 0.0:
        if abs_angle >= float(far_side_angle_deg):
            return (
                "far_left",
                "LOOK FAR LEFT - scan 45-90 deg left, high uncertainty, front/back ambiguous",
                67.5,
                90.0,
            )
        return (
            "left",
            f"LOOK LEFT - approx {abs_angle:.0f} deg from center, scan +/-{scan_half:.0f} deg, front/back ambiguous",
            angle,
            sector,
        )
    if abs_angle >= float(far_side_angle_deg):
        return (
            "far_right",
            "LOOK FAR RIGHT - scan 45-90 deg right, high uncertainty, front/back ambiguous",
            -67.5,
            90.0,
        )
    return (
        "right",
        f"LOOK RIGHT - approx {abs_angle:.0f} deg from center, scan +/-{scan_half:.0f} deg, front/back ambiguous",
        angle,
        sector,
    )


def _possible_azimuths(front_heading_deg: float | None, angle_from_center_deg: float) -> tuple[float | None, float | None]:
    if front_heading_deg is None:
        return None, None
    heading = float(front_heading_deg)
    angle = float(angle_from_center_deg)
    return (heading + angle) % 360.0, (heading + 180.0 - angle) % 360.0


def _unknown_result(reason: str, sector_width_deg: float) -> TwoMicDirectionResult:
    return TwoMicDirectionResult(
        side="uncertain",
        reason=reason,
        look_label="unknown",
        look_hint="DIRECTION UNKNOWN - scan left and right, front/back ambiguous",
        sector_width_deg=float(sector_width_deg),
        front_back_ambiguous=True,
        stable=False,
    )
