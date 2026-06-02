from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from station.harmonic import harmonic_score


@dataclass
class StationDetectorStateConfig:
    f0_min: int = 500
    f0_max: int = 2200
    max_freq: int = 7000
    min_harmonics: int = 3
    min_suspect_threshold: float = 14.0
    min_alert_threshold: float = 22.0
    calibration_seconds: float = 8.0
    min_alert_duration_sec: float = 3.0
    clear_after_sec: float = 2.5


@dataclass
class ChannelEvidence:
    channel_index: int
    rms: Optional[float]
    harmonic_score: float
    best_f0_hz: Optional[int]
    passed: bool


@dataclass
class StationDetectionFrame:
    status: str
    confidence: float
    harmonic_score: float
    best_f0_hz: Optional[int]
    duration_sec: float
    rms: float
    peak: float
    calibrated: bool
    per_channel: list[ChannelEvidence] = field(default_factory=list)
    strongest_channel: Optional[int] = None
    agreement_count: int = 0
    channel_count: int = 1
    suspect_threshold: float = 14.0
    alert_threshold: float = 22.0


def _sigmoid_score(score: float, threshold: float) -> float:
    return float(1 / (1 + np.exp(-(score - threshold) / 3.0)))


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    audio64 = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(audio64 * audio64)))


def _as_samples_by_channels(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio.reshape(-1, 1)
    if audio.ndim != 2:
        raise ValueError(f"audio must be mono or 2D multi-channel, got shape {audio.shape}")
    return audio


class StationDetectorState:
    def __init__(self, config: StationDetectorStateConfig):
        self.config = config
        self.calibration_started_at: Optional[float] = None
        self.calibrated = False
        self.background_scores: list[float] = []
        self.suspect_threshold = float(config.min_suspect_threshold)
        self.alert_threshold = float(config.min_alert_threshold)
        self._evidence_started_at: Optional[float] = None
        self._last_evidence_at: Optional[float] = None
        self._last_active_status: Optional[str] = None

    def update(
        self,
        audio: np.ndarray,
        sr: int,
        timestamp: float,
        hf_p_drone: Optional[float] = None,
        cnn_p_drone: Optional[float] = None,
    ) -> StationDetectionFrame:
        channels = _as_samples_by_channels(audio)
        mono = channels.mean(axis=1)
        mono_rms = _rms(mono)
        mono_peak = float(np.max(np.abs(mono))) if mono.size else 0.0

        mono_score, mono_f0, _ = harmonic_score(
            mono,
            sr,
            self.config.f0_min,
            self.config.f0_max,
            self.config.max_freq,
            self.config.min_harmonics,
        )
        mono_score = float(mono_score)

        if self.calibration_started_at is None:
            self.calibration_started_at = timestamp

        if not self.calibrated:
            self.background_scores.append(mono_score)
            elapsed = timestamp - self.calibration_started_at
            if elapsed < self.config.calibration_seconds:
                return self._frame(
                    status="calibrating",
                    confidence=_sigmoid_score(mono_score, self.config.min_suspect_threshold),
                    harmonic_score=mono_score,
                    best_f0_hz=mono_f0,
                    duration_sec=0.0,
                    rms=mono_rms,
                    peak=mono_peak,
                    per_channel=self._channel_evidence(channels, sr, self.config.min_suspect_threshold),
                    channel_count=channels.shape[1],
                )
            self._finish_calibration()

        per_channel = self._channel_evidence(channels, sr, self.suspect_threshold)
        strongest = max(per_channel, key=lambda item: item.harmonic_score, default=None)
        strongest_score = strongest.harmonic_score if strongest is not None else 0.0
        evidence_score = max(mono_score, strongest_score)
        best_f0 = mono_f0 if mono_score >= strongest_score else (strongest.best_f0_hz if strongest else mono_f0)
        agreement_count = sum(1 for item in per_channel if item.passed)

        has_suspect_harmonic = evidence_score >= self.suspect_threshold
        has_alert_harmonic = evidence_score >= self.alert_threshold

        status, duration = self._status_for_evidence(
            timestamp=timestamp,
            has_suspect_harmonic=has_suspect_harmonic,
            has_alert_harmonic=has_alert_harmonic,
        )
        confidence = self._confidence(evidence_score, has_suspect_harmonic, hf_p_drone, cnn_p_drone, agreement_count)

        return self._frame(
            status=status,
            confidence=confidence,
            harmonic_score=mono_score,
            best_f0_hz=best_f0,
            duration_sec=duration,
            rms=mono_rms,
            peak=mono_peak,
            per_channel=per_channel,
            strongest_channel=strongest.channel_index if strongest is not None else None,
            agreement_count=agreement_count,
            channel_count=channels.shape[1],
        )

    def _finish_calibration(self) -> None:
        scores = np.asarray(self.background_scores, dtype=np.float64)
        if scores.size:
            median = float(np.median(scores))
            mad = float(np.median(np.abs(scores - median)))
            p95 = float(np.percentile(scores, 95))
            suspect = max(self.config.min_suspect_threshold, p95 + max(3.0 * mad, 1.0))
            alert = max(self.config.min_alert_threshold, p95 + max(6.0 * mad, 3.0), suspect + 4.0)
        else:
            suspect = self.config.min_suspect_threshold
            alert = self.config.min_alert_threshold

        self.suspect_threshold = float(suspect)
        self.alert_threshold = float(alert)
        self.calibrated = True

    def _channel_evidence(self, channels: np.ndarray, sr: int, threshold: float) -> list[ChannelEvidence]:
        evidence = []
        for idx in range(channels.shape[1]):
            channel = channels[:, idx]
            score, best_f0, _ = harmonic_score(
                channel,
                sr,
                self.config.f0_min,
                self.config.f0_max,
                self.config.max_freq,
                self.config.min_harmonics,
            )
            score = float(score)
            evidence.append(
                ChannelEvidence(
                    channel_index=idx,
                    rms=_rms(channel),
                    harmonic_score=score,
                    best_f0_hz=best_f0,
                    passed=score >= threshold,
                )
            )
        return evidence

    def _status_for_evidence(
        self,
        timestamp: float,
        has_suspect_harmonic: bool,
        has_alert_harmonic: bool,
    ) -> tuple[str, float]:
        if has_suspect_harmonic:
            if self._evidence_started_at is None:
                self._evidence_started_at = timestamp
            self._last_evidence_at = timestamp
            duration = max(0.0, timestamp - self._evidence_started_at)

            if has_alert_harmonic and duration >= self.config.min_alert_duration_sec:
                status = "alert"
            elif has_alert_harmonic:
                status = "drone_like"
            else:
                status = "suspect"

            self._last_active_status = status
            return status, duration

        if self._last_evidence_at is not None:
            clean_for = timestamp - self._last_evidence_at
            if clean_for < self.config.clear_after_sec and self._last_active_status is not None:
                duration = 0.0
                if self._evidence_started_at is not None:
                    duration = max(0.0, self._last_evidence_at - self._evidence_started_at)
                return self._last_active_status, duration

        self._evidence_started_at = None
        self._last_evidence_at = None
        self._last_active_status = None
        return "background", 0.0

    def _confidence(
        self,
        evidence_score: float,
        has_suspect_harmonic: bool,
        hf_p_drone: Optional[float],
        cnn_p_drone: Optional[float],
        agreement_count: int,
    ) -> float:
        confidence = _sigmoid_score(evidence_score, self.suspect_threshold)
        if has_suspect_harmonic:
            support = max(float(hf_p_drone or 0.0), float(cnn_p_drone or 0.0))
            confidence = min(1.0, confidence + 0.15 * support)
            if agreement_count >= 2:
                confidence = min(1.0, confidence + 0.08)
        return float(confidence)

    def _frame(
        self,
        status: str,
        confidence: float,
        harmonic_score: float,
        best_f0_hz: Optional[int],
        duration_sec: float,
        rms: float,
        peak: float,
        per_channel: list[ChannelEvidence],
        channel_count: int,
        strongest_channel: Optional[int] = None,
        agreement_count: int = 0,
    ) -> StationDetectionFrame:
        return StationDetectionFrame(
            status=status,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            harmonic_score=float(harmonic_score),
            best_f0_hz=best_f0_hz,
            duration_sec=float(duration_sec),
            rms=float(rms),
            peak=float(peak),
            calibrated=self.calibrated,
            per_channel=per_channel,
            strongest_channel=strongest_channel,
            agreement_count=int(agreement_count),
            channel_count=int(channel_count),
            suspect_threshold=self.suspect_threshold,
            alert_threshold=self.alert_threshold,
        )
