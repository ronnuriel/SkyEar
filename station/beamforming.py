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
    bearing_stable: bool = False
    bearing_uncertainty_deg: float | None = None


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
    best_score = float(scores[best_idx])
    median_score = float(np.nanmedian(scores))
    spread = float(np.nanstd(scores) + 1e-12)
    snr_gain_db = 10.0 * np.log10((best_score + 1e-12) / (median_score + 1e-12)) if best_score > 0 and median_score > 0 else 0.0
    close = angles[scores >= best_score - spread]
    uncertainty = float(max(scan_step_deg, len(close) * scan_step_deg / 2.0))
    return BeamformingResult(
        bearing_deg=best_angle,
        beamforming_method=str(method),
        beam_score=best_score,
        beam_snr_gain_db=float(snr_gain_db),
        bearing_stable=uncertainty <= float(bearing_stability_deg),
        bearing_uncertainty_deg=uncertainty,
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


def _srp_phat_score(audio: np.ndarray, sr: int, positions: np.ndarray, bearing_deg: float) -> float:
    direction = direction_vector(float(bearing_deg))
    delays = positions @ direction / SPEED_OF_SOUND
    score = 0.0
    pairs = 0
    for left in range(audio.shape[1]):
        for right in range(left + 1, audio.shape[1]):
            tau = float(delays[right] - delays[left])
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
