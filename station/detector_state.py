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
    ml_strong_threshold: float = 0.90
    ml_candidate_harmonic_min_pct: float = 0.15
    ml_drone_like_harmonic_min_pct: float = 0.45
    ml_only_duration_for_drone_like_sec: float = 8.0
    smoothing_enabled: bool = True
    harmonic_smoothing_windows: int = 5
    harmonic_smoothing_method: str = "median"
    alert_enter_pct: float = 0.85
    alert_exit_pct: float = 0.55
    drone_like_enter_pct: float = 0.45
    drone_like_exit_pct: float = 0.25
    min_alert_windows: int = 2
    min_drone_like_windows: int = 2
    f0_family_tolerance_hz: float = 140.0


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
    harmonic_evidence_pct_raw: float = 0.0
    harmonic_evidence_pct_smoothed: float = 0.0
    harmonic_score_smoothed: float = 0.0
    ml_drone_pct: Optional[float] = None
    ml_drone_pct_smoothed: Optional[float] = None
    hf_negative: bool = False
    hf_positive: bool = False
    decision_reason: str = "no acoustic evidence"
    operator_label: str = "background"
    raw_best_f0_hz: Optional[int] = None
    canonical_best_f0_hz: Optional[int] = None
    f0_family_stable: bool = False


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
        self.raw_harmonic_score_history: list[float] = []
        self.harmonic_evidence_pct_history: list[float] = []
        self.hf_p_history: list[Optional[float]] = []
        self.raw_f0_history: list[Optional[int]] = []
        self.operator_state_history: list[str] = []
        self._canonical_f0_history: list[Optional[int]] = []
        self._alert_below_since: Optional[float] = None

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

        raw_best_f0 = best_f0
        canonical_best_f0 = self._canonical_f0_family(raw_best_f0)
        harmonic_evidence_pct_raw = self._harmonic_evidence_pct(evidence_score)
        ml_drone_pct = hf_p_drone if hf_p_drone is not None else cnn_p_drone
        self._record_window_history(evidence_score, harmonic_evidence_pct_raw, hf_p_drone, raw_best_f0, canonical_best_f0)
        harmonic_score_smoothed = self._smoothed(self.raw_harmonic_score_history)
        harmonic_evidence_pct_smoothed = self._smoothed(self.harmonic_evidence_pct_history)
        ml_drone_pct_smoothed = ml_drone_pct
        has_suspect_harmonic = harmonic_evidence_pct_raw > 0.0 or evidence_score >= self.suspect_threshold
        self._record_history(canonical_best_f0, harmonic_score_smoothed)
        f0_stable = self._f0_is_stable()
        f0_family_stable = self._f0_family_is_stable()
        harmonic_evidence_pct = harmonic_evidence_pct_raw
        hf_available = hf_p_drone is not None
        hf_positive = hf_available and float(hf_p_drone) >= self.config.advisory_threshold
        hf_negative = hf_available and float(hf_p_drone) < self.config.hf_negative_threshold
        ml_strong = ml_drone_pct_smoothed is not None and float(ml_drone_pct_smoothed) >= self.config.ml_strong_threshold
        advisory_support = hf_positive or float(cnn_p_drone or 0.0) >= self.config.advisory_threshold
        strong_channel_agreement = agreement_count >= 2
        single_channel = channels.shape[1] == 1
        multi_channel = channels.shape[1] > 1
        negative_hf_veto = self.config.hf_negative_caps_status and hf_negative
        strong_multichannel_evidence = multi_channel and strong_channel_agreement and f0_family_stable

        if single_channel:
            alert_ready = f0_family_stable and not negative_hf_veto
            if self.config.hf_required_for_single_channel_alert and hf_available:
                alert_ready = alert_ready and hf_positive
            drone_like_ready = not negative_hf_veto and (hf_positive or f0_family_stable)
        else:
            alert_ready = strong_channel_agreement or f0_family_stable
            drone_like_ready = advisory_support or f0_family_stable or strong_channel_agreement
            if negative_hf_veto:
                alert_ready = strong_multichannel_evidence
                drone_like_ready = strong_multichannel_evidence

        status, duration = self._status_for_smoothed_evidence(
            timestamp=timestamp,
            has_suspect_harmonic=has_suspect_harmonic,
            harmonic_evidence_pct_smoothed=harmonic_evidence_pct_smoothed,
            ml_strong=ml_strong,
            hf_negative=hf_negative,
            drone_like_ready=drone_like_ready,
            alert_ready=alert_ready,
        )
        if ml_strong and not hf_negative and status == "background":
            status = "suspect"
            self._last_active_status = status
        decision_harmonic_pct = harmonic_evidence_pct_smoothed
        if hf_negative:
            decision_harmonic_pct = max(harmonic_evidence_pct_raw, harmonic_evidence_pct_smoothed)
        operator_label = self._operator_label(
            status=status,
            harmonic_evidence_pct=decision_harmonic_pct,
            hf_negative=hf_negative,
            ml_strong=ml_strong,
        )
        decision_reason = self._decision_reason(
            status=status,
            has_suspect_harmonic=has_suspect_harmonic,
            harmonic_evidence_pct=decision_harmonic_pct,
            hf_negative=hf_negative,
            hf_positive=hf_positive,
            ml_strong=ml_strong,
            strong_channel_agreement=strong_channel_agreement,
            f0_stable=f0_family_stable,
        )
        self._status_history.append(status)
        self._trim_history(self._status_history)
        self.operator_state_history.append(operator_label)
        self._trim_history(self.operator_state_history)
        confidence = self._confidence(evidence_score, has_suspect_harmonic, hf_p_drone, cnn_p_drone, agreement_count)

        return self._frame(
            status=status,
            confidence=confidence,
            harmonic_score=mono_score,
            harmonic_score_smoothed=harmonic_score_smoothed,
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
            harmonic_evidence_pct_raw=harmonic_evidence_pct_raw,
            harmonic_evidence_pct_smoothed=harmonic_evidence_pct_smoothed,
            ml_drone_pct=ml_drone_pct,
            ml_drone_pct_smoothed=ml_drone_pct_smoothed,
            hf_negative=hf_negative,
            hf_positive=hf_positive,
            decision_reason=decision_reason,
            operator_label=operator_label,
            raw_best_f0_hz=raw_best_f0,
            canonical_best_f0_hz=canonical_best_f0,
            f0_family_stable=f0_family_stable,
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

    def _status_for_smoothed_evidence(
        self,
        timestamp: float,
        has_suspect_harmonic: bool,
        harmonic_evidence_pct_smoothed: float,
        ml_strong: bool,
        hf_negative: bool,
        drone_like_ready: bool,
        alert_ready: bool,
    ) -> tuple[str, float]:
        has_operator_signal = has_suspect_harmonic or ml_strong
        if has_operator_signal:
            if self._evidence_started_at is None:
                self._evidence_started_at = timestamp
            self._last_evidence_at = timestamp
            duration = max(0.0, timestamp - self._evidence_started_at)

            if self._last_active_status == "alert":
                if harmonic_evidence_pct_smoothed < self.config.alert_exit_pct:
                    if self._alert_below_since is None:
                        self._alert_below_since = timestamp
                    if timestamp - self._alert_below_since < self.config.clear_after_sec:
                        return "alert", duration
                else:
                    self._alert_below_since = None
                    return "alert", duration
            else:
                self._alert_below_since = None

            alert_enter_ready = (
                harmonic_evidence_pct_smoothed >= self.config.alert_enter_pct
                and self._recent_pct_all(self.config.alert_enter_pct, self.config.min_alert_windows)
                and duration >= self.config.min_alert_duration_sec
                and alert_ready
            )
            if alert_enter_ready:
                self._last_active_status = "alert"
                self._alert_below_since = None
                return "alert", duration

            if hf_negative and harmonic_evidence_pct_smoothed >= self.config.drone_like_enter_pct and not alert_ready:
                self._last_active_status = "suspect"
                return "suspect", duration

            drone_like_enter_ready = (
                harmonic_evidence_pct_smoothed >= self.config.drone_like_enter_pct
                and self._recent_pct_all(self.config.drone_like_enter_pct, self.config.min_drone_like_windows)
                and drone_like_ready
            )
            if drone_like_enter_ready:
                self._last_active_status = "drone_like"
                return "drone_like", duration

            if (
                self._last_active_status == "drone_like"
                and harmonic_evidence_pct_smoothed >= self.config.drone_like_exit_pct
                and not hf_negative
            ):
                return "drone_like", duration

            self._last_active_status = "suspect"
            return "suspect", duration

        if self._last_evidence_at is not None:
            clean_for = timestamp - self._last_evidence_at
            if clean_for < self.config.clear_after_sec and self._last_active_status is not None:
                duration = 0.0
                if self._evidence_started_at is not None:
                    duration = max(0.0, self._last_evidence_at - self._evidence_started_at)
                if self._last_active_status == "alert" and harmonic_evidence_pct_smoothed >= self.config.alert_exit_pct:
                    return "alert", duration
                if self._last_active_status == "drone_like" and harmonic_evidence_pct_smoothed >= self.config.drone_like_exit_pct:
                    return "drone_like", duration
                return "suspect", duration

        self._evidence_started_at = None
        self._last_evidence_at = None
        self._last_active_status = None
        self._alert_below_since = None
        return "background", 0.0

    def _harmonic_evidence_pct(self, evidence_score: float) -> float:
        denominator = max(self.alert_threshold - self.suspect_threshold, 1e-6)
        value = (float(evidence_score) - self.suspect_threshold) / denominator
        return float(np.clip(value, 0.0, 1.0))

    def _record_window_history(
        self,
        harmonic_score: float,
        harmonic_evidence_pct: float,
        hf_p_drone: Optional[float],
        raw_f0_hz: Optional[int],
        canonical_f0_hz: Optional[int],
    ) -> None:
        self.raw_harmonic_score_history.append(float(harmonic_score))
        self.harmonic_evidence_pct_history.append(float(harmonic_evidence_pct))
        self.hf_p_history.append(None if hf_p_drone is None else float(hf_p_drone))
        self.raw_f0_history.append(raw_f0_hz)
        self._canonical_f0_history.append(canonical_f0_hz)
        self._trim_history(self.raw_harmonic_score_history)
        self._trim_history(self.harmonic_evidence_pct_history)
        self._trim_history(self.hf_p_history)
        self._trim_history(self.raw_f0_history)
        self._trim_history(self._canonical_f0_history)

    def _smoothed(self, values: list[float]) -> float:
        if not values:
            return 0.0
        window = values[-max(1, int(self.config.harmonic_smoothing_windows)) :]
        if not self.config.smoothing_enabled:
            return float(window[-1])
        if self.config.harmonic_smoothing_method == "mean":
            return float(np.mean(np.asarray(window, dtype=np.float64)))
        return float(np.median(np.asarray(window, dtype=np.float64)))

    def _recent_pct_all(self, threshold: float, min_windows: int) -> bool:
        min_windows = max(1, int(min_windows))
        if len(self.harmonic_evidence_pct_history) < min_windows:
            return False
        return all(value >= threshold for value in self.harmonic_evidence_pct_history[-min_windows:])

    def _same_f0_family(self, left: float, right: float) -> bool:
        tolerance = float(self.config.f0_family_tolerance_hz)
        return (
            abs(left - right) <= tolerance
            or abs(left * 2.0 - right) <= tolerance
            or abs(left - right * 2.0) <= tolerance
        )

    def _canonical_f0_family(self, raw_f0_hz: Optional[int]) -> Optional[int]:
        if raw_f0_hz is None:
            return None
        raw = float(raw_f0_hz)
        for previous in reversed(self._canonical_f0_history):
            if previous is not None and self._same_f0_family(raw, float(previous)):
                return int(previous)
        return int(raw_f0_hz)

    def _f0_family_is_stable(self) -> bool:
        if not self.config.stability_enabled:
            return True
        usable = [
            f0
            for f0, pct in zip(self._canonical_f0_history, self.harmonic_evidence_pct_history)
            if f0 is not None and pct >= self.config.drone_like_exit_pct
        ]
        if len(usable) < self.config.stability_min_score_windows:
            return False
        first = float(usable[-1])
        recent = usable[-self.config.stability_min_score_windows :]
        return all(self._same_f0_family(float(item), first) for item in recent)

    def _decision_reason(
        self,
        status: str,
        has_suspect_harmonic: bool,
        harmonic_evidence_pct: float,
        hf_negative: bool,
        hf_positive: bool,
        ml_strong: bool,
        strong_channel_agreement: bool,
        f0_stable: bool,
    ) -> str:
        high_harmonic = harmonic_evidence_pct >= 0.75
        if hf_negative and high_harmonic:
            return "harmonic source detected, but ML strongly rejects drone"
        if ml_strong and harmonic_evidence_pct < max(0.30, self.config.ml_candidate_harmonic_min_pct):
            return "ML strongly indicates drone; harmonic rotor evidence is weak"
        if ml_strong and harmonic_evidence_pct < self.config.ml_drone_like_harmonic_min_pct:
            return "ML strongly indicates drone with partial harmonic support"
        if (hf_positive or ml_strong) and harmonic_evidence_pct >= self.config.ml_drone_like_harmonic_min_pct:
            return "ML and harmonic rotor evidence agree"
        if hf_positive and not has_suspect_harmonic:
            return "ML-only suspect; harmonic rotor evidence is weak"
        if has_suspect_harmonic and status == "suspect":
            return "harmonic source detected; awaiting stronger drone evidence"
        if strong_channel_agreement and f0_stable:
            return "multi-channel stable harmonic evidence"
        return "background or insufficient evidence"

    def _operator_label(
        self,
        status: str,
        harmonic_evidence_pct: float,
        hf_negative: bool,
        ml_strong: bool,
    ) -> str:
        if status == "alert":
            return "alert"
        if hf_negative and harmonic_evidence_pct >= 0.75:
            return "non_drone_harmonic"
        if status == "drone_like":
            return "drone_like"
        if ml_strong:
            return "ml_drone_candidate"
        return "background"

    def _record_history(self, best_f0_hz: Optional[int], score: float) -> None:
        self._best_f0_history.append(best_f0_hz)
        self._score_history.append(float(score))
        self._trim_history(self._best_f0_history)
        self._trim_history(self._score_history)

    def _trim_history(self, values: list) -> None:
        max_items = max(
            1,
            int(self.config.stability_history_windows),
            int(self.config.harmonic_smoothing_windows),
            int(self.config.min_alert_windows),
            int(self.config.min_drone_like_windows),
        )
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
        harmonic_evidence_pct_raw: float = 0.0,
        harmonic_evidence_pct_smoothed: float = 0.0,
        harmonic_score_smoothed: Optional[float] = None,
        ml_drone_pct: Optional[float] = None,
        ml_drone_pct_smoothed: Optional[float] = None,
        hf_negative: bool = False,
        hf_positive: bool = False,
        decision_reason: str = "no acoustic evidence",
        operator_label: str = "background",
        raw_best_f0_hz: Optional[int] = None,
        canonical_best_f0_hz: Optional[int] = None,
        f0_family_stable: bool = False,
    ) -> StationDetectionFrame:
        harmonic_pct = float(np.clip(harmonic_evidence_pct, 0.0, 1.0))
        raw_pct = float(np.clip(harmonic_evidence_pct_raw, 0.0, 1.0))
        smoothed_pct = float(np.clip(harmonic_evidence_pct_smoothed, 0.0, 1.0))
        return StationDetectionFrame(
            status=status,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            harmonic_score=float(harmonic_score),
            harmonic_score_smoothed=float(harmonic_score if harmonic_score_smoothed is None else harmonic_score_smoothed),
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
            harmonic_evidence_pct=harmonic_pct,
            harmonic_evidence_pct_raw=raw_pct,
            harmonic_evidence_pct_smoothed=smoothed_pct,
            ml_drone_pct=None if ml_drone_pct is None else float(np.clip(ml_drone_pct, 0.0, 1.0)),
            ml_drone_pct_smoothed=None
            if ml_drone_pct_smoothed is None
            else float(np.clip(ml_drone_pct_smoothed, 0.0, 1.0)),
            hf_negative=bool(hf_negative),
            hf_positive=bool(hf_positive),
            decision_reason=str(decision_reason),
            operator_label=str(operator_label),
            raw_best_f0_hz=raw_best_f0_hz if raw_best_f0_hz is not None else best_f0_hz,
            canonical_best_f0_hz=canonical_best_f0_hz if canonical_best_f0_hz is not None else best_f0_hz,
            f0_family_stable=bool(f0_family_stable),
        )
