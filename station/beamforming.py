from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from station.direction import SPEED_OF_SOUND, bandpass, direction_vector, fractional_delay


BeamformingMethod = Literal["delay_and_sum", "srp_phat", "gcc_phat"]


@dataclass
class BeamformingResult:
    bearing_deg: float | None = None
    beamforming_method: str = "delay_and_sum"
    beam_score: float | None = None
    beam_snr_gain_db: float | None = None
    beam_confidence_pct: float | None = None
    beam_peak_to_median: float | None = None
    beam_peak_to_second_peak: float | None = None
    bearing_stable: bool = False
    bearing_uncertainty_deg: float | None = None
    beam_scan_deg: list[float] | None = None
    beam_scan_score: list[float] | None = None


def delay_and_sum_beam(audio: np.ndarray, sr: int, mic_positions_m: np.ndarray, bearing_deg: float) -> np.ndarray:
    audio = _as_multichannel(audio)
    positions = np.asarray(mic_positions_m, dtype=np.float64)
    direction = direction_vector(float(bearing_deg))
    delays_sec = positions @ direction / SPEED_OF_SOUND
    delays_sec -= np.mean(delays_sec)
    aligned = [fractional_delay(audio[:, idx], -float(delays_sec[idx]) * int(sr)) for idx in range(audio.shape[1])]
    return np.mean(np.stack(aligned, axis=1), axis=1)


def estimate_bearing(
    audio: np.ndarray,
    sr: int,
    mic_positions_m: np.ndarray,
    method: BeamformingMethod = "delay_and_sum",
    scan_step_deg: int = 5,
    low_hz: int = 500,
    high_hz: int = 7000,
    bearing_stability_deg: float = 15.0,
    include_scan: bool = False,
) -> BeamformingResult:
    audio = _as_multichannel(audio)
    positions = np.asarray(mic_positions_m, dtype=np.float64)
    if audio.shape[1] < 2 or positions.shape[0] != audio.shape[1]:
        return BeamformingResult(beamforming_method=str(method))

    try:
        filtered = bandpass(audio, int(sr), low=int(low_hz), high=int(high_hz))
    except Exception:
        filtered = audio

    angles = np.arange(0.0, 360.0, float(max(1, scan_step_deg)))
    normalized_method = str(method).lower().replace("-", "_")
    if normalized_method in {"srp_phat", "gcc_phat"}:
        scores = np.asarray([_srp_phat_score(filtered, int(sr), positions, angle) for angle in angles])
        method = normalized_method
    else:
        scores = np.asarray([_delay_sum_score(filtered, int(sr), positions, angle) for angle in angles])
        method = "delay_and_sum"

    if scores.size == 0 or not np.isfinite(scores).any():
        return BeamformingResult(beamforming_method=str(method))
    best_idx = int(np.nanargmax(scores))
    best_angle = float(angles[best_idx])
    finite_scores = np.asarray([score for score in scores if np.isfinite(score)], dtype=np.float64)
    best_score = float(scores[best_idx])
    median_score = float(np.nanmedian(finite_scores))
    spread = float(np.nanstd(finite_scores) + 1e-12)
    sorted_scores = np.sort(finite_scores)
    second_score = float(sorted_scores[-2]) if sorted_scores.size >= 2 else best_score
    peak_to_median = float(abs(best_score) / (abs(median_score) + 1e-12))
    peak_to_second = float(abs(best_score) / (abs(second_score) + 1e-12))
    snr_gain_db = 10.0 * np.log10(peak_to_median) if peak_to_median > 0 else 0.0
    confidence = _beam_confidence(best_score, median_score, second_score, spread)
    close = angles[scores >= best_score - max(spread, abs(best_score - median_score) * 0.35)]
    uncertainty = float(max(scan_step_deg, len(close) * scan_step_deg / 2.0))
    return BeamformingResult(
        bearing_deg=best_angle,
        beamforming_method=str(method),
        beam_score=best_score,
        beam_snr_gain_db=float(snr_gain_db),
        beam_confidence_pct=confidence,
        beam_peak_to_median=peak_to_median,
        beam_peak_to_second_peak=peak_to_second,
        bearing_stable=uncertainty <= float(bearing_stability_deg) and confidence >= 0.35,
        bearing_uncertainty_deg=uncertainty,
        beam_scan_deg=[float(angle) for angle in angles] if include_scan else None,
        beam_scan_score=[float(score) for score in scores] if include_scan else None,
    )


def _as_multichannel(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio.reshape(-1, 1)
    if audio.ndim != 2:
        raise ValueError(f"audio must be mono or 2D multi-channel, got shape {audio.shape}")
    return audio


def _delay_sum_score(audio: np.ndarray, sr: int, positions: np.ndarray, bearing_deg: float) -> float:
    beam = delay_and_sum_beam(audio, sr, positions, bearing_deg)
    return float(np.mean(beam * beam))


def _beam_confidence(best_score: float, median_score: float, second_score: float, spread: float) -> float:
    if best_score <= 0 or spread <= 0:
        return 0.0
    median_margin = max(0.0, (best_score - median_score) / (abs(best_score) + 1e-12))
    second_margin = max(0.0, (best_score - second_score) / (abs(best_score) + 1e-12))
    sharpness = max(0.0, (best_score - median_score) / (spread + 1e-12))
    confidence = 0.55 * median_margin + 0.25 * min(1.0, sharpness / 4.0) + 0.20 * min(1.0, second_margin * 4.0)
    return float(np.clip(confidence, 0.0, 1.0))


def _srp_phat_score(audio: np.ndarray, sr: int, positions: np.ndarray, bearing_deg: float) -> float:
    direction = direction_vector(float(bearing_deg))
    delays = positions @ direction / SPEED_OF_SOUND
    score = 0.0
    pairs = 0
    for left in range(audio.shape[1]):
        for right in range(left + 1, audio.shape[1]):
            tau = float(delays[left] - delays[right])
            score += _gcc_phat_at_tau(audio[:, left], audio[:, right], sr, tau)
            pairs += 1
    return float(score / max(1, pairs))


def _gcc_phat_at_tau(left: np.ndarray, right: np.ndarray, sr: int, tau_sec: float) -> float:
    n = int(2 ** np.ceil(np.log2(max(len(left), len(right)) * 2)))
    left_fft = np.fft.rfft(left, n=n)
    right_fft = np.fft.rfft(right, n=n)
    cross = left_fft * np.conj(right_fft)
    cross /= np.abs(cross) + 1e-12
    corr = np.fft.irfft(cross, n=n)
    corr = np.concatenate((corr[-(n // 2) :], corr[: n // 2]))
    center = n // 2
    idx = int(round(center + tau_sec * sr))
    if idx < 0 or idx >= corr.size:
        return 0.0
    return float(corr[idx])
