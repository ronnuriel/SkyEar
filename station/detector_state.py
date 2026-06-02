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
    stability_enabled: bool = True
    stability_history_windows: int = 4
    stability_max_f0_std_hz: float = 80.0
    stability_min_score_windows: int = 3
    advisory_threshold: float = 0.70
    hf_negative_threshold: float = 0.20
    hf_required_for_single_channel_alert: bool = True
    hf_negative_caps_status: bool = True


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
    f0_stable: bool = False
    harmonic_evidence_pct: float = 0.0
    ml_drone_pct: Optional[float] = None
    hf_negative: bool = False
    hf_positive: bool = False
    decision_reason: str = "no acoustic evidence"


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
        self._best_f0_history: list[Optional[int]] = []
        self._score_history: list[float] = []
        self._status_history: list[str] = []

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
        self._record_history(best_f0, evidence_score)
        f0_stable = self._f0_is_stable()
        harmonic_evidence_pct = self._harmonic_evidence_pct(evidence_score)
        ml_drone_pct = hf_p_drone if hf_p_drone is not None else cnn_p_drone
        hf_available = hf_p_drone is not None
        hf_positive = hf_available and float(hf_p_drone) >= self.config.advisory_threshold
        hf_negative = hf_available and float(hf_p_drone) < self.config.hf_negative_threshold
        advisory_support = hf_positive or float(cnn_p_drone or 0.0) >= self.config.advisory_threshold
        strong_channel_agreement = agreement_count >= 2
        single_channel = channels.shape[1] == 1
        multi_channel = channels.shape[1] > 1
        negative_hf_veto = self.config.hf_negative_caps_status and hf_negative
        strong_multichannel_evidence = multi_channel and strong_channel_agreement and f0_stable

        if single_channel:
            alert_ready = f0_stable and not negative_hf_veto
            if self.config.hf_required_for_single_channel_alert and hf_available:
                alert_ready = alert_ready and hf_positive
            drone_like_ready = not negative_hf_veto and (hf_positive or f0_stable)
        else:
            alert_ready = strong_channel_agreement or f0_stable
            drone_like_ready = advisory_support or f0_stable or strong_channel_agreement
            if negative_hf_veto:
                alert_ready = strong_multichannel_evidence
                drone_like_ready = strong_multichannel_evidence

        status, duration = self._status_for_evidence(
            timestamp=timestamp,
            has_suspect_harmonic=has_suspect_harmonic,
            has_alert_harmonic=has_alert_harmonic,
            drone_like_ready=drone_like_ready,
            alert_ready=alert_ready,
        )
        decision_reason = self._decision_reason(
            status=status,
            has_suspect_harmonic=has_suspect_harmonic,
            harmonic_evidence_pct=harmonic_evidence_pct,
            hf_negative=hf_negative,
            hf_positive=hf_positive,
            strong_channel_agreement=strong_channel_agreement,
            f0_stable=f0_stable,
        )
        self._status_history.append(status)
        self._trim_history(self._status_history)
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
            f0_stable=f0_stable,
            harmonic_evidence_pct=harmonic_evidence_pct,
            ml_drone_pct=ml_drone_pct,
            hf_negative=hf_negative,
            hf_positive=hf_positive,
            decision_reason=decision_reason,
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
        drone_like_ready: bool,
        alert_ready: bool,
    ) -> tuple[str, float]:
        if has_suspect_harmonic:
            if self._evidence_started_at is None:
                self._evidence_started_at = timestamp
            self._last_evidence_at = timestamp
            duration = max(0.0, timestamp - self._evidence_started_at)

            if has_alert_harmonic and duration >= self.config.min_alert_duration_sec and alert_ready:
                status = "alert"
            elif has_alert_harmonic and drone_like_ready:
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

    def _harmonic_evidence_pct(self, evidence_score: float) -> float:
        denominator = max(self.alert_threshold - self.suspect_threshold, 1e-6)
        value = (float(evidence_score) - self.suspect_threshold) / denominator
        return float(np.clip(value, 0.0, 1.0))

    def _decision_reason(
        self,
        status: str,
        has_suspect_harmonic: bool,
        harmonic_evidence_pct: float,
        hf_negative: bool,
        hf_positive: bool,
        strong_channel_agreement: bool,
        f0_stable: bool,
    ) -> str:
        high_harmonic = harmonic_evidence_pct >= 0.75
        if hf_negative and high_harmonic:
            return "harmonic source detected, but ML strongly rejects drone"
        if hf_positive and high_harmonic:
            return "ML and harmonic rotor evidence agree"
        if hf_positive and not has_suspect_harmonic:
            return "ML-only suspect; harmonic rotor evidence is weak"
        if has_suspect_harmonic and status == "suspect":
            return "harmonic source detected; awaiting stronger drone evidence"
        if strong_channel_agreement and f0_stable:
            return "multi-channel stable harmonic evidence"
        return "background or insufficient evidence"

    def _record_history(self, best_f0_hz: Optional[int], score: float) -> None:
        self._best_f0_history.append(best_f0_hz)
        self._score_history.append(float(score))
        self._trim_history(self._best_f0_history)
        self._trim_history(self._score_history)

    def _trim_history(self, values: list) -> None:
        max_items = max(1, int(self.config.stability_history_windows))
        del values[:-max_items]

    def _f0_is_stable(self) -> bool:
        if not self.config.stability_enabled:
            return True
        usable = [
            f0
            for f0, score in zip(self._best_f0_history, self._score_history)
            if f0 is not None and score >= self.suspect_threshold
        ]
        if len(usable) < self.config.stability_min_score_windows:
            return False
        return float(np.std(np.asarray(usable, dtype=np.float64))) <= self.config.stability_max_f0_std_hz

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
        f0_stable: bool = False,
        harmonic_evidence_pct: float = 0.0,
        ml_drone_pct: Optional[float] = None,
        hf_negative: bool = False,
        hf_positive: bool = False,
        decision_reason: str = "no acoustic evidence",
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
            f0_stable=bool(f0_stable),
            harmonic_evidence_pct=float(np.clip(harmonic_evidence_pct, 0.0, 1.0)),
            ml_drone_pct=None if ml_drone_pct is None else float(np.clip(ml_drone_pct, 0.0, 1.0)),
            hf_negative=bool(hf_negative),
            hf_positive=bool(hf_positive),
            decision_reason=str(decision_reason),
        )
