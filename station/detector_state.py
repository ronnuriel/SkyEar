from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from station.harmonic import harmonic_score


@dataclass
class StationDetectorStateConfig:
    detection_profile: str = "conservative"
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
    hf_watch_threshold: float = 0.50
    hf_candidate_threshold: float = 0.70
    hf_strong_threshold: float = 0.85
    ml_positive_threshold: float = 0.90
    single_mic_candidate_run_required: int = 2
    single_mic_strong_run_required: int = 3
    allow_single_mic_alert: bool = False
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
    min_ml_candidate_windows: int = 2
    min_ml_drone_like_windows: int = 3
    ml_spike_single_window_caps_to_candidate: bool = True
    ml_strong_recent_window_sec: float = 3.0
    f0_family_tolerance_hz: float = 140.0
    max_hf_age_sec: float = 6.0
    max_acoustic_age_sec: float = 6.0
    harmonic_lock_enabled: bool = True
    harmonic_lock_min_duration_sec: float = 3.0
    harmonic_lock_hold_sec: float = 5.0
    harmonic_f0_jump_penalty: float = 0.5
    harmonic_ridge_max_drift_hz: float = 80.0
    harmonic_track_bandwidth_hz: float = 120.0
    harmonic_noise_floor_rolling_median_sec: float = 10.0


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
    combined_drone_evidence_pct: float = 0.0
    hf_negative: bool = False
    hf_positive: bool = False
    hf_error: bool = False
    harmonic_activity_duration_sec: float = 0.0
    decision_reason: str = "no acoustic evidence"
    operator_label: str = "background"
    candidate_run: int = 0
    hf_candidate_run: int = 0
    acoustic_candidate_run: int = 0
    fused_candidate_run: int = 0
    ml_positive_run: int = 0
    strong_run: int = 0
    hf_age_sec: Optional[float] = None
    harmonic_age_sec: Optional[float] = None
    max_hf_age_sec: float = 6.0
    max_acoustic_age_sec: float = 6.0
    estimated_detection_delay_sec: Optional[float] = None
    raw_best_f0_hz: Optional[int] = None
    canonical_best_f0_hz: Optional[int] = None
    f0_family_stable: bool = False
    decision_stage: str = "background"
    blocked_by: str = ""
    hf_watch_threshold: float = 0.50
    hf_candidate_threshold: float = 0.70
    hf_strong_threshold: float = 0.85
    hf_candidate_pass: bool = False
    hf_strong_pass: bool = False
    harmonic_pass: bool = False
    single_channel_mode: bool = True
    candidate_block_reason: str = ""
    alert_block_reason: str = ""
    alert_blocked_reason: str = ""
    why_candidate_run_reset: str = ""
    harmonic_track_active: bool = False
    tracked_f0_hz: Optional[int] = None
    tracked_ridges: list[dict[str, float | int]] = field(default_factory=list)
    harmonic_track_age_sec: float = 0.0
    f0_raw_hz: Optional[int] = None
    f0_track_hz: Optional[int] = None
    f0_jump_reason: str = ""
    stable_harmonic_ridge_count: int = 0
    longest_ridge_duration_sec: float = 0.0


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


def _combined_drone_evidence(ml_drone_pct: Optional[float], harmonic_evidence_pct: float) -> float:
    ml = float(np.clip(float(ml_drone_pct or 0.0), 0.0, 1.0))
    harmonic = float(np.clip(float(harmonic_evidence_pct or 0.0), 0.0, 1.0))
    return float(np.clip((2.0 * ml * harmonic) / (ml + harmonic + 1e-6), 0.0, 1.0))


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
        self.ml_watch_history: list[bool] = []
        self.ml_candidate_history: list[bool] = []
        self.ml_strong_history: list[bool] = []
        self.acoustic_candidate_history: list[bool] = []
        self.fused_candidate_history: list[bool] = []
        self.combined_strong_history: list[bool] = []
        self.harmonic_partial_history: list[bool] = []
        self.strong_candidate_history: list[bool] = []
        self.ml_window_timestamp_history: list[float] = []
        self._last_hf_at: Optional[float] = None
        self._last_acoustic_at: Optional[float] = None
        self._harmonic_track_started_at: Optional[float] = None
        self._harmonic_track_last_seen_at: Optional[float] = None
        self._tracked_f0_hz: Optional[int] = None
        self._longest_ridge_duration_sec: float = 0.0
        self._alert_below_since: Optional[float] = None

    def _profile(self) -> str:
        profile = str(getattr(self.config, "detection_profile", "conservative") or "conservative").strip().lower()
        return profile if profile in {"conservative", "field_debug"} else "conservative"

    def _effective_watch_threshold(self) -> float:
        if self._profile() == "field_debug":
            return float(self.config.hf_watch_threshold)
        return float(self.config.ml_positive_threshold)

    def _effective_candidate_threshold(self) -> float:
        if self._profile() == "field_debug":
            return float(self.config.hf_candidate_threshold)
        return float(self.config.ml_positive_threshold)

    def _effective_strong_threshold(self) -> float:
        if self._profile() == "field_debug":
            return float(self.config.hf_strong_threshold)
        return float(max(self.config.ml_positive_threshold, self.config.ml_strong_threshold))

    def _threshold_passes(self, ml_drone_pct: Optional[float]) -> tuple[bool, bool, bool]:
        if ml_drone_pct is None:
            return False, False, False
        ml = float(ml_drone_pct)
        return (
            ml >= self._effective_watch_threshold(),
            ml >= self._effective_candidate_threshold(),
            ml >= self._effective_strong_threshold(),
        )

    def update(
        self,
        audio: np.ndarray,
        sr: int,
        timestamp: float,
        hf_p_drone: Optional[float] = None,
        cnn_p_drone: Optional[float] = None,
        hf_error: bool = False,
        hf_age_sec: Optional[float] = None,
        acoustic_bearing_support: bool = False,
    ) -> StationDetectionFrame:
        channels = _as_samples_by_channels(audio)
        mono = channels.mean(axis=1)
        mono_rms = _rms(mono)
        mono_peak = float(np.max(np.abs(mono))) if mono.size else 0.0

        mono_score, mono_f0, raw_mono_f0, track_meta = self._tracked_harmonic_score(mono, sr, timestamp)
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

        raw_best_f0 = raw_mono_f0 if mono_score >= strongest_score else best_f0
        canonical_best_f0 = self._canonical_f0_family(raw_best_f0)
        harmonic_evidence_pct_raw = self._harmonic_evidence_pct(evidence_score)
        ml_drone_pct = hf_p_drone if hf_p_drone is not None else cnn_p_drone
        ml_drone_pct_smoothed = ml_drone_pct
        if ml_drone_pct is not None:
            self._last_hf_at = float(timestamp) - float(hf_age_sec or 0.0)
        effective_hf_age_sec = (
            None
            if self._last_hf_at is None
            else max(0.0, float(timestamp) - float(self._last_hf_at))
        )
        hf_watch_pass, hf_candidate_pass, hf_strong_pass = self._threshold_passes(ml_drone_pct_smoothed)
        hf_stale = effective_hf_age_sec is not None and effective_hf_age_sec > float(self.config.max_hf_age_sec)
        if hf_stale:
            hf_watch_pass = hf_candidate_pass = hf_strong_pass = False
        ml_strong = hf_strong_pass
        self._record_window_history(
            evidence_score,
            harmonic_evidence_pct_raw,
            hf_p_drone,
            raw_best_f0,
            canonical_best_f0,
            timestamp=timestamp,
            ml_watch=hf_watch_pass,
            ml_candidate=hf_candidate_pass,
            ml_strong=ml_strong,
        )
        harmonic_score_smoothed = self._smoothed(self.raw_harmonic_score_history)
        harmonic_evidence_pct_smoothed = self._smoothed(self.harmonic_evidence_pct_history)
        combined_drone_evidence_pct = _combined_drone_evidence(
            ml_drone_pct_smoothed,
            harmonic_evidence_pct_smoothed,
        )
        self._update_current_ml_support(
            combined_drone_evidence_pct=combined_drone_evidence_pct,
            harmonic_evidence_pct_smoothed=harmonic_evidence_pct_smoothed,
        )
        has_suspect_harmonic = harmonic_evidence_pct_raw > 0.0 or evidence_score >= self.suspect_threshold
        if has_suspect_harmonic or harmonic_evidence_pct_smoothed > 0.0:
            self._last_acoustic_at = float(timestamp)
        harmonic_age_sec = (
            None
            if self._last_acoustic_at is None
            else max(0.0, float(timestamp) - float(self._last_acoustic_at))
        )
        harmonic_stale = harmonic_age_sec is not None and harmonic_age_sec > float(self.config.max_acoustic_age_sec)
        if harmonic_stale:
            harmonic_evidence_pct_smoothed = 0.0
            combined_drone_evidence_pct = _combined_drone_evidence(ml_drone_pct_smoothed, harmonic_evidence_pct_smoothed)
        track_state = self._update_harmonic_track(
            timestamp=timestamp,
            raw_f0_hz=raw_mono_f0,
            selected_f0_hz=best_f0,
            harmonic_evidence_pct=harmonic_evidence_pct_raw,
            harmonic_evidence_pct_smoothed=harmonic_evidence_pct_smoothed,
            details=track_meta.get("details") or [],
            f0_jump_reason=str(track_meta.get("f0_jump_reason") or ""),
        )
        self._record_history(canonical_best_f0, harmonic_score_smoothed)
        f0_stable = self._f0_is_stable()
        f0_family_stable = self._f0_family_is_stable()
        harmonic_evidence_pct = harmonic_evidence_pct_raw
        hf_available = hf_p_drone is not None
        hf_positive = hf_available and float(hf_p_drone) >= self.config.advisory_threshold
        hf_negative = hf_available and float(hf_p_drone) < self.config.hf_negative_threshold
        ml_unavailable = bool(hf_error) or ml_drone_pct_smoothed is None
        acoustic_support = bool(
            not harmonic_stale
            and (
                (has_suspect_harmonic and harmonic_evidence_pct_smoothed >= self.config.ml_candidate_harmonic_min_pct)
                or bool(acoustic_bearing_support)
            )
        )
        harmonic_pass = acoustic_support
        self._update_current_candidate_support(
            acoustic_support=acoustic_support,
            fused_support=bool(hf_candidate_pass and acoustic_support and combined_drone_evidence_pct > 0.0),
            strong_support=bool(hf_strong_pass and acoustic_support and combined_drone_evidence_pct >= 0.35),
        )
        ml_candidate_persistent = self._ml_candidate_persistent()
        strong_local_candidate = self._strong_local_candidate_persistent()
        ml_drone_like_persistent = self._ml_drone_like_persistent(
            f0_family_stable=f0_family_stable,
            harmonic_evidence_pct_smoothed=harmonic_evidence_pct_smoothed,
        )
        hf_candidate_run = self._current_true_run(self.ml_candidate_history)
        acoustic_candidate_run = self._current_true_run(self.acoustic_candidate_history)
        fused_candidate_run = self._current_true_run(self.fused_candidate_history)
        ml_positive_run = hf_candidate_run
        candidate_run = fused_candidate_run
        strong_run = self._current_true_run(self.strong_candidate_history)
        estimated_detection_delay_sec = self._current_run_elapsed_sec(self.fused_candidate_history)
        advisory_support = hf_positive or float(cnn_p_drone or 0.0) >= self.config.advisory_threshold
        meaningful_channel_agreement = channels.shape[1] >= 2 and agreement_count >= 2
        strong_channel_agreement = (
            channels.shape[1] >= 4
            and agreement_count >= max(2, int(np.ceil(channels.shape[1] * 0.5)))
        )
        single_channel = channels.shape[1] == 1
        multi_channel = channels.shape[1] > 1
        negative_hf_veto = self.config.hf_negative_caps_status and hf_negative
        strong_multichannel_evidence = multi_channel and strong_channel_agreement and f0_family_stable and acoustic_support
        fused_alert_evidence = (
            not ml_unavailable
            and not hf_stale
            and not harmonic_stale
            and hf_strong_pass
            and acoustic_support
            and combined_drone_evidence_pct >= self.config.alert_enter_pct
            and fused_candidate_run >= max(1, int(self.config.min_alert_windows))
        )
        alert_blocked_reason = ""

        if single_channel:
            alert_ready = bool(self.config.allow_single_mic_alert) and not negative_hf_veto and fused_alert_evidence and (
                f0_family_stable or f0_stable or track_state["harmonic_track_active"]
            )
            if not bool(self.config.allow_single_mic_alert):
                alert_blocked_reason = "single_mic_alert_disabled"
            elif ml_unavailable:
                alert_blocked_reason = "ml_unavailable"
            elif hf_stale:
                alert_blocked_reason = "hf_stale"
            elif harmonic_stale:
                alert_blocked_reason = "acoustic_stale"
            elif negative_hf_veto:
                alert_blocked_reason = "hf_negative"
            elif not hf_strong_pass:
                alert_blocked_reason = "hf_below_strong_threshold"
            elif not acoustic_support:
                alert_blocked_reason = "acoustic_below_candidate_support"
            elif fused_candidate_run < max(1, int(self.config.min_alert_windows)):
                alert_blocked_reason = "fused_run_below_required"
            elif combined_drone_evidence_pct < self.config.alert_enter_pct:
                alert_blocked_reason = "combined_below_alert_threshold"
            elif not (f0_family_stable or f0_stable or track_state["harmonic_track_active"]):
                alert_blocked_reason = "f0_not_stable"
            drone_like_ready = not ml_unavailable and not negative_hf_veto and ml_drone_like_persistent
        else:
            alert_ready = fused_alert_evidence and (
                strong_multichannel_evidence or f0_family_stable or f0_stable or track_state["harmonic_track_active"]
            )
            drone_like_ready = (
                not ml_unavailable
                and not hf_stale
                and not harmonic_stale
                and acoustic_support
                and (advisory_support or ml_drone_like_persistent or strong_multichannel_evidence)
            )
            if negative_hf_veto:
                alert_ready = False
                drone_like_ready = False
            if not alert_ready:
                if ml_unavailable and not strong_multichannel_evidence:
                    alert_blocked_reason = "ml_unavailable_without_strong_multichannel"
                elif hf_stale:
                    alert_blocked_reason = "hf_stale"
                elif harmonic_stale:
                    alert_blocked_reason = "acoustic_stale"
                elif negative_hf_veto and not strong_multichannel_evidence:
                    alert_blocked_reason = "hf_negative_without_strong_multichannel"
                elif not acoustic_support:
                    alert_blocked_reason = "acoustic_below_candidate_support"
                elif fused_candidate_run < max(1, int(self.config.min_alert_windows)):
                    alert_blocked_reason = "fused_run_below_required"

        status, duration = self._status_for_smoothed_evidence(
            timestamp=timestamp,
            has_suspect_harmonic=has_suspect_harmonic,
            harmonic_evidence_pct_smoothed=harmonic_evidence_pct_smoothed,
            combined_drone_evidence_pct=combined_drone_evidence_pct,
            ml_strong=ml_strong,
            ml_drone_like_persistent=ml_drone_like_persistent,
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
        if (
            not hf_negative
            and status != "alert"
            and (
                combined_drone_evidence_pct >= 0.60
                or (ml_strong and acoustic_support and decision_harmonic_pct >= 0.40)
            )
        ):
            if ml_drone_like_persistent:
                status = "drone_like"
                self._last_active_status = status
            elif self.config.ml_spike_single_window_caps_to_candidate:
                status = "suspect"
                self._last_active_status = status
        if (
            status in {"alert", "drone_like"}
            and not strong_multichannel_evidence
            and (ml_unavailable or candidate_run == 0)
        ):
            status = "suspect" if has_suspect_harmonic or decision_harmonic_pct > 0.0 else "background"
            self._last_active_status = status if status != "background" else None
        operator_label = self._operator_label(
            status=status,
            harmonic_evidence_pct=decision_harmonic_pct,
            combined_drone_evidence_pct=combined_drone_evidence_pct,
            hf_negative=hf_negative,
            ml_unavailable=ml_unavailable,
            hf_watch_pass=hf_watch_pass,
            hf_candidate_pass=hf_candidate_pass,
            hf_strong_pass=hf_strong_pass,
            harmonic_pass=harmonic_pass,
            single_channel=single_channel,
            ml_strong=ml_strong,
            ml_candidate_persistent=fused_candidate_run >= max(1, int(self.config.min_ml_candidate_windows)),
            strong_local_candidate=strong_local_candidate,
            ml_drone_like_persistent=ml_drone_like_persistent,
        )
        status, operator_label = self._enforce_status_label_consistency(
            status=status,
            operator_label=operator_label,
            has_suspect_harmonic=has_suspect_harmonic,
            harmonic_evidence_pct=decision_harmonic_pct,
            ml_drone_pct=ml_drone_pct_smoothed,
            combined_drone_evidence_pct=combined_drone_evidence_pct,
            hf_p_drone=hf_p_drone,
            hf_error=bool(hf_error),
            channel_count=channels.shape[1],
            agreement_count=agreement_count,
            f0_family_stable=f0_family_stable,
            ml_drone_like_persistent=ml_drone_like_persistent,
            candidate_run=candidate_run,
        )
        decision_reason = self._decision_reason(
            status=status,
            has_suspect_harmonic=has_suspect_harmonic,
            harmonic_evidence_pct=decision_harmonic_pct,
            combined_drone_evidence_pct=combined_drone_evidence_pct,
            hf_negative=hf_negative,
            hf_positive=hf_positive,
            ml_unavailable=ml_unavailable,
            ml_strong=ml_strong,
            strong_channel_agreement=meaningful_channel_agreement,
            f0_stable=f0_family_stable,
        )
        if self._profile() == "field_debug" and single_channel and not bool(self.config.allow_single_mic_alert):
            if status in {"alert", "drone_like"}:
                status = "suspect" if has_suspect_harmonic or harmonic_pass else "background"
                if harmonic_pass and strong_local_candidate:
                    operator_label = "strong_local_candidate"
                elif harmonic_pass and fused_candidate_run >= max(1, int(self.config.single_mic_candidate_run_required)):
                    operator_label = "local_drone_candidate"
                elif harmonic_pass and hf_candidate_pass:
                    operator_label = "weak_local_candidate"
                elif harmonic_pass and hf_watch_pass:
                    operator_label = "acoustic_drone_watch"
                else:
                    operator_label = "acoustic_harmonic_source" if has_suspect_harmonic else "background"
                alert_blocked_reason = alert_blocked_reason or "single_mic_alert_disabled"

        candidate_block_reason = ""
        if ml_drone_pct_smoothed is None:
            candidate_block_reason = "ml_unavailable"
        elif not hf_candidate_pass:
            candidate_block_reason = "hf_below_candidate_threshold"
        elif self._profile() == "field_debug" and not harmonic_pass:
            candidate_block_reason = "harmonic_below_candidate_support"
        elif hf_candidate_pass and not harmonic_pass:
            candidate_block_reason = "acoustic_below_candidate_support"

        why_candidate_run_reset = ""
        if candidate_run == 0 and not hf_candidate_pass:
            why_candidate_run_reset = candidate_block_reason
        elif hf_candidate_pass and not harmonic_pass:
            why_candidate_run_reset = candidate_block_reason or "acoustic_below_candidate_support"

        if (
            status == "background"
            and (
                hf_candidate_run > 0
                or operator_label
                in {
                    "acoustic_drone_watch",
                    "weak_local_candidate",
                    "local_drone_candidate",
                    "strong_local_candidate",
                }
            )
        ):
            status = "suspect"
            self._last_active_status = status

        blocked_reasons: list[str] = []
        if alert_blocked_reason:
            blocked_reasons.append(alert_blocked_reason)
        if candidate_block_reason:
            blocked_reasons.append(candidate_block_reason)
        if hf_negative:
            blocked_reasons.append("hf_negative")
        if ml_unavailable:
            blocked_reasons.append("ml_unavailable")
        if single_channel and not bool(self.config.allow_single_mic_alert):
            blocked_reasons.append("single_channel_alert_disabled")
        blocked_by = ",".join(dict.fromkeys(reason for reason in blocked_reasons if reason))

        if operator_label in {"alert", "drone_like", "strong_local_candidate", "local_drone_candidate", "weak_local_candidate", "acoustic_drone_watch", "ml_drone_candidate", "non_drone_harmonic", "acoustic_harmonic_source"}:
            decision_stage = operator_label
        elif status == "suspect":
            decision_stage = "suspect"
        else:
            decision_stage = "background"
        self._status_history.append(status)
        self._trim_history(self._status_history)
        self.operator_state_history.append(operator_label)
        self._trim_history(self.operator_state_history)
        confidence = self._confidence(
            evidence_score,
            has_suspect_harmonic,
            hf_p_drone,
            cnn_p_drone,
            agreement_count,
            channels.shape[1],
        )

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
            combined_drone_evidence_pct=combined_drone_evidence_pct,
            hf_negative=hf_negative,
            hf_positive=hf_positive,
            hf_error=bool(hf_error),
            harmonic_activity_duration_sec=duration if has_suspect_harmonic else 0.0,
            decision_reason=decision_reason,
            operator_label=operator_label,
            candidate_run=candidate_run,
            hf_candidate_run=hf_candidate_run,
            acoustic_candidate_run=acoustic_candidate_run,
            fused_candidate_run=fused_candidate_run,
            ml_positive_run=ml_positive_run,
            strong_run=strong_run,
            hf_age_sec=effective_hf_age_sec,
            harmonic_age_sec=harmonic_age_sec,
            max_hf_age_sec=self.config.max_hf_age_sec,
            max_acoustic_age_sec=self.config.max_acoustic_age_sec,
            estimated_detection_delay_sec=estimated_detection_delay_sec,
            raw_best_f0_hz=raw_best_f0,
            canonical_best_f0_hz=canonical_best_f0,
            harmonic_track_active=track_state["harmonic_track_active"],
            tracked_f0_hz=track_state["tracked_f0_hz"],
            tracked_ridges=track_state["tracked_ridges"],
            harmonic_track_age_sec=track_state["harmonic_track_age_sec"],
            f0_raw_hz=raw_mono_f0,
            f0_track_hz=track_state["tracked_f0_hz"],
            f0_jump_reason=track_state["f0_jump_reason"],
            stable_harmonic_ridge_count=track_state["stable_harmonic_ridge_count"],
            longest_ridge_duration_sec=track_state["longest_ridge_duration_sec"],
            f0_family_stable=f0_family_stable,
            decision_stage=decision_stage,
            blocked_by=blocked_by,
            hf_watch_threshold=self._effective_watch_threshold(),
            hf_candidate_threshold=self._effective_candidate_threshold(),
            hf_strong_threshold=self._effective_strong_threshold(),
            hf_candidate_pass=hf_candidate_pass,
            hf_strong_pass=hf_strong_pass,
            harmonic_pass=harmonic_pass,
            single_channel_mode=single_channel,
            candidate_block_reason=candidate_block_reason,
            alert_block_reason=alert_blocked_reason,
            alert_blocked_reason=alert_blocked_reason,
            why_candidate_run_reset=why_candidate_run_reset,
        )

    def _tracked_harmonic_score(
        self,
        mono: np.ndarray,
        sr: int,
        timestamp: float,
    ) -> tuple[float, Optional[int], Optional[int], dict[str, object]]:
        broad_min = 300 if self.config.harmonic_lock_enabled else self.config.f0_min
        broad_max = 3500 if self.config.harmonic_lock_enabled else self.config.f0_max
        raw_score, raw_f0, raw_details = harmonic_score(
            mono,
            sr,
            broad_min,
            broad_max,
            self.config.max_freq,
            self.config.min_harmonics,
        )
        if not self.config.harmonic_lock_enabled or self._tracked_f0_hz is None:
            return float(raw_score), raw_f0, raw_f0, {
                "details": raw_details,
                "f0_jump_reason": "unlocked_broad_scan" if self.config.harmonic_lock_enabled else "",
            }

        bandwidth = float(self.config.harmonic_track_bandwidth_hz)
        low = max(100, int(round(float(self._tracked_f0_hz) - bandwidth)))
        high = min(int(3500), int(round(float(self._tracked_f0_hz) + bandwidth)))
        track_score, track_f0, track_details = harmonic_score(
            mono,
            sr,
            low,
            high,
            self.config.max_freq,
            self.config.min_harmonics,
        )
        if track_f0 is not None and raw_f0 is not None:
            jump_hz = abs(float(raw_f0) - float(self._tracked_f0_hz))
            raw_same_family = self._same_f0_family(float(raw_f0), float(self._tracked_f0_hz))
            hold_track = (
                not raw_same_family
                and jump_hz > float(self.config.harmonic_ridge_max_drift_hz)
                and float(track_score) >= float(raw_score) * float(self.config.harmonic_f0_jump_penalty)
            )
            if hold_track:
                return float(track_score), track_f0, raw_f0, {
                    "details": track_details,
                    "f0_jump_reason": f"held_track_penalized_raw_jump_{jump_hz:.0f}hz",
                }
        if track_f0 is not None and raw_f0 is None:
            return float(track_score), track_f0, None, {
                "details": track_details,
                "f0_jump_reason": "held_track_missing_raw_f0",
            }
        return float(raw_score), raw_f0, raw_f0, {
            "details": raw_details,
            "f0_jump_reason": "",
        }

    def _update_harmonic_track(
        self,
        *,
        timestamp: float,
        raw_f0_hz: Optional[int],
        selected_f0_hz: Optional[int],
        harmonic_evidence_pct: float,
        harmonic_evidence_pct_smoothed: float,
        details: list[float],
        f0_jump_reason: str,
    ) -> dict[str, object]:
        if not self.config.harmonic_lock_enabled:
            return self._harmonic_track_state(False, raw_f0_hz, selected_f0_hz, [], 0.0, f0_jump_reason)

        has_ridge = selected_f0_hz is not None and (
            float(harmonic_evidence_pct) > 0.0 or float(harmonic_evidence_pct_smoothed) > 0.0
        )
        if has_ridge:
            if self._tracked_f0_hz is None or not self._same_f0_family(float(selected_f0_hz), float(self._tracked_f0_hz)):
                self._tracked_f0_hz = int(selected_f0_hz)
                self._harmonic_track_started_at = float(timestamp)
            elif abs(float(selected_f0_hz) - float(self._tracked_f0_hz)) <= float(self.config.harmonic_ridge_max_drift_hz):
                self._tracked_f0_hz = int(round(0.75 * float(self._tracked_f0_hz) + 0.25 * float(selected_f0_hz)))
            self._harmonic_track_last_seen_at = float(timestamp)
        elif (
            self._harmonic_track_last_seen_at is not None
            and float(timestamp) - float(self._harmonic_track_last_seen_at) > float(self.config.harmonic_lock_hold_sec)
        ):
            self._tracked_f0_hz = None
            self._harmonic_track_started_at = None
            self._harmonic_track_last_seen_at = None

        age = 0.0
        if self._harmonic_track_started_at is not None and self._harmonic_track_last_seen_at is not None:
            age = max(0.0, float(timestamp) - float(self._harmonic_track_started_at))
        active = bool(
            self._tracked_f0_hz is not None
            and self._harmonic_track_last_seen_at is not None
            and float(timestamp) - float(self._harmonic_track_last_seen_at) <= float(self.config.harmonic_lock_hold_sec)
            and age >= float(self.config.harmonic_lock_min_duration_sec)
        )
        if active:
            self._longest_ridge_duration_sec = max(self._longest_ridge_duration_sec, age)
        ridges = self._tracked_ridges(self._tracked_f0_hz, details)
        return self._harmonic_track_state(active, raw_f0_hz, self._tracked_f0_hz, ridges, age, f0_jump_reason)

    def _tracked_ridges(self, f0_hz: Optional[int], details: list[float]) -> list[dict[str, float | int]]:
        if f0_hz is None:
            return []
        count = max(0, len(details))
        return [
            {"k": k, "freq_hz": float(k * float(f0_hz))}
            for k in range(1, count + 1)
            if k * float(f0_hz) <= float(self.config.max_freq)
        ]

    def _harmonic_track_state(
        self,
        active: bool,
        raw_f0_hz: Optional[int],
        tracked_f0_hz: Optional[int],
        ridges: list[dict[str, float | int]],
        age_sec: float,
        jump_reason: str,
    ) -> dict[str, object]:
        return {
            "harmonic_track_active": bool(active),
            "tracked_f0_hz": tracked_f0_hz,
            "tracked_ridges": ridges,
            "harmonic_track_age_sec": float(max(0.0, age_sec)),
            "f0_raw_hz": raw_f0_hz,
            "f0_track_hz": tracked_f0_hz,
            "f0_jump_reason": str(jump_reason or ""),
            "stable_harmonic_ridge_count": len(ridges) if active else 0,
            "longest_ridge_duration_sec": float(self._longest_ridge_duration_sec),
        }

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
        combined_drone_evidence_pct: float,
        ml_strong: bool,
        ml_drone_like_persistent: bool,
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
                (
                    harmonic_evidence_pct_smoothed >= self.config.alert_enter_pct
                    and self._recent_pct_all(self.config.alert_enter_pct, self.config.min_alert_windows)
                    and (not ml_strong or ml_drone_like_persistent)
                )
                or (combined_drone_evidence_pct >= 0.75 and ml_drone_like_persistent)
            ) and duration >= self.config.min_alert_duration_sec and alert_ready
            if alert_enter_ready:
                self._last_active_status = "alert"
                self._alert_below_since = None
                return "alert", duration

            if hf_negative and harmonic_evidence_pct_smoothed >= self.config.drone_like_enter_pct and not alert_ready:
                self._last_active_status = "suspect"
                return "suspect", duration

            harmonic_drone_like_enter_ready = (
                harmonic_evidence_pct_smoothed >= self.config.drone_like_enter_pct
                and self._recent_pct_all(self.config.drone_like_enter_pct, self.config.min_drone_like_windows)
            )
            if ml_strong and not ml_drone_like_persistent:
                harmonic_drone_like_enter_ready = False
            drone_like_enter_ready = harmonic_drone_like_enter_ready or (
                combined_drone_evidence_pct >= 0.60 and ml_drone_like_persistent
            )
            if drone_like_enter_ready and drone_like_ready:
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
        timestamp: Optional[float] = None,
        ml_watch: bool = False,
        ml_candidate: bool = False,
        ml_strong: bool = False,
    ) -> None:
        self.raw_harmonic_score_history.append(float(harmonic_score))
        self.harmonic_evidence_pct_history.append(float(harmonic_evidence_pct))
        self.hf_p_history.append(None if hf_p_drone is None else float(hf_p_drone))
        self.raw_f0_history.append(raw_f0_hz)
        self._canonical_f0_history.append(canonical_f0_hz)
        self.ml_watch_history.append(bool(ml_watch))
        self.ml_candidate_history.append(bool(ml_candidate))
        self.ml_strong_history.append(bool(ml_strong))
        self.acoustic_candidate_history.append(False)
        self.fused_candidate_history.append(False)
        self.combined_strong_history.append(False)
        self.harmonic_partial_history.append(False)
        self.strong_candidate_history.append(False)
        self.ml_window_timestamp_history.append(0.0 if timestamp is None else float(timestamp))
        self._trim_history(self.raw_harmonic_score_history)
        self._trim_history(self.harmonic_evidence_pct_history)
        self._trim_history(self.hf_p_history)
        self._trim_history(self.raw_f0_history)
        self._trim_history(self._canonical_f0_history)
        self._trim_history(self.ml_watch_history)
        self._trim_history(self.ml_candidate_history)
        self._trim_history(self.ml_strong_history)
        self._trim_history(self.acoustic_candidate_history)
        self._trim_history(self.fused_candidate_history)
        self._trim_history(self.combined_strong_history)
        self._trim_history(self.harmonic_partial_history)
        self._trim_history(self.strong_candidate_history)
        self._trim_history(self.ml_window_timestamp_history)

    def _update_current_ml_support(self, combined_drone_evidence_pct: float, harmonic_evidence_pct_smoothed: float) -> None:
        if self.combined_strong_history:
            self.combined_strong_history[-1] = float(combined_drone_evidence_pct) >= 0.35
        if self.harmonic_partial_history:
            self.harmonic_partial_history[-1] = float(harmonic_evidence_pct_smoothed) >= 0.25

    def _update_current_candidate_support(
        self,
        *,
        acoustic_support: bool,
        fused_support: bool,
        strong_support: bool,
    ) -> None:
        if self.acoustic_candidate_history:
            self.acoustic_candidate_history[-1] = bool(acoustic_support)
        if self.fused_candidate_history:
            self.fused_candidate_history[-1] = bool(fused_support)
        if self.strong_candidate_history:
            self.strong_candidate_history[-1] = bool(strong_support)

    def _ml_candidate_persistent(self) -> bool:
        required = max(1, int(self.config.single_mic_candidate_run_required if self._profile() == "field_debug" else self.config.min_ml_candidate_windows))
        recent_window_count = max(3, required)
        if len(self.ml_candidate_history) < required:
            return False
        return sum(1 for value in self.ml_candidate_history[-recent_window_count:] if value) >= required

    def _strong_local_candidate_persistent(self) -> bool:
        required = max(1, int(self.config.single_mic_strong_run_required if self._profile() == "field_debug" else self.config.min_ml_drone_like_windows))
        recent_window_count = max(5, required)
        if len(self.strong_candidate_history) < required:
            return False
        return sum(1 for value in self.strong_candidate_history[-recent_window_count:] if value) >= required

    def _ml_drone_like_persistent(self, f0_family_stable: bool, harmonic_evidence_pct_smoothed: float) -> bool:
        ml_required = max(1, int(self.config.min_ml_drone_like_windows))
        recent_window_count = max(5, ml_required)
        history = self.ml_strong_history if self._profile() != "field_debug" else self.ml_candidate_history
        if len(history) < ml_required:
            return False
        ml_count = sum(1 for value in history[-recent_window_count:] if value)
        has_recent_combined_support = any(self.combined_strong_history[-recent_window_count:])
        has_harmonic_support = float(harmonic_evidence_pct_smoothed) >= 0.25 or any(
            self.harmonic_partial_history[-recent_window_count:]
        )
        return ml_count >= ml_required and (
            has_recent_combined_support or has_harmonic_support or f0_family_stable
        )

    def _current_true_run(self, values: list[bool]) -> int:
        run = 0
        for value in reversed(values):
            if not value:
                break
            run += 1
        return run

    def _current_run_elapsed_sec(self, values: list[bool]) -> Optional[float]:
        run = self._current_true_run(values)
        if run <= 0 or len(self.ml_window_timestamp_history) < run:
            return None
        return max(0.0, float(self.ml_window_timestamp_history[-1]) - float(self.ml_window_timestamp_history[-run]))

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
        combined_drone_evidence_pct: float,
        hf_negative: bool,
        hf_positive: bool,
        ml_unavailable: bool,
        ml_strong: bool,
        strong_channel_agreement: bool,
        f0_stable: bool,
    ) -> str:
        high_harmonic = harmonic_evidence_pct >= 0.75
        if ml_unavailable and harmonic_evidence_pct > 0.0:
            return "HF unavailable — harmonic-only mode, alert disabled"
        if hf_negative and high_harmonic:
            return "harmonic source detected, but ML strongly rejects drone"
        if combined_drone_evidence_pct >= 0.75:
            return "strong combined ML and harmonic rotor evidence"
        if combined_drone_evidence_pct >= 0.60:
            return "ML and harmonic rotor evidence strongly agree"
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
        combined_drone_evidence_pct: float,
        hf_negative: bool,
        ml_unavailable: bool,
        hf_watch_pass: bool,
        hf_candidate_pass: bool,
        hf_strong_pass: bool,
        harmonic_pass: bool,
        single_channel: bool,
        ml_strong: bool,
        ml_candidate_persistent: bool,
        strong_local_candidate: bool,
        ml_drone_like_persistent: bool,
    ) -> str:
        if status == "alert":
            return "alert"
        if ml_unavailable and harmonic_evidence_pct > 0.0:
            return "acoustic_harmonic_source"
        if self._profile() == "field_debug" and single_channel:
            if hf_negative and harmonic_pass:
                return "non_drone_harmonic" if harmonic_evidence_pct >= 0.75 else "acoustic_harmonic_source"
            if harmonic_pass and hf_strong_pass and strong_local_candidate:
                return "strong_local_candidate"
            if harmonic_pass and hf_candidate_pass and ml_candidate_persistent:
                return "local_drone_candidate"
            if harmonic_pass and hf_candidate_pass:
                return "weak_local_candidate"
            if harmonic_pass and hf_watch_pass:
                return "acoustic_drone_watch"
            if harmonic_pass:
                return "acoustic_harmonic_source"
            if hf_candidate_pass:
                return "ml_drone_candidate"
            if harmonic_evidence_pct > 0.0:
                return "acoustic_harmonic_source"
            return "background"
        if hf_negative and harmonic_evidence_pct >= 0.75:
            return "non_drone_harmonic"
        if status == "drone_like":
            return "drone_like" if (ml_drone_like_persistent or not ml_strong) else "ml_drone_candidate"
        if combined_drone_evidence_pct >= 0.60 and ml_drone_like_persistent:
            return "drone_like"
        if strong_local_candidate:
            return "strong_local_candidate"
        if ml_candidate_persistent:
            return "local_drone_candidate"
        if ml_strong or ml_candidate_persistent:
            return "ml_drone_candidate"
        return "background"

    def _enforce_status_label_consistency(
        self,
        status: str,
        operator_label: str,
        has_suspect_harmonic: bool,
        harmonic_evidence_pct: float,
        ml_drone_pct: Optional[float],
        combined_drone_evidence_pct: float,
        hf_p_drone: Optional[float],
        channel_count: int,
        agreement_count: int,
        f0_family_stable: bool,
        hf_error: bool = False,
        ml_drone_like_persistent: bool = True,
        candidate_run: int = 0,
    ) -> tuple[str, str]:
        strong_multichannel_evidence = (
            channel_count >= 4
            and agreement_count >= max(2, int(np.ceil(channel_count * 0.5)))
            and f0_family_stable
        )
        hf_negative = hf_p_drone is not None and float(hf_p_drone) < self.config.hf_negative_threshold
        ml_unavailable = bool(hf_error) or ml_drone_pct is None
        ml_low = ml_drone_pct is not None and float(ml_drone_pct) < self.config.advisory_threshold

        if operator_label == "background" and status in {"drone_like", "alert"}:
            status = "suspect" if has_suspect_harmonic else "background"

        if operator_label in {"non_drone_harmonic", "acoustic_harmonic_source"} and status in {"drone_like", "alert"}:
            status = "suspect"

        if (
            self._profile() == "field_debug"
            and channel_count == 1
            and not bool(self.config.allow_single_mic_alert)
            and operator_label
            in {
                "acoustic_drone_watch",
                "weak_local_candidate",
                "local_drone_candidate",
                "strong_local_candidate",
            }
            and status in {"drone_like", "alert"}
        ):
            status = "suspect"

        if (
            status in {"drone_like", "alert"}
            and not strong_multichannel_evidence
            and (ml_unavailable or candidate_run == 0)
        ):
            status = "suspect" if harmonic_evidence_pct > 0.0 or has_suspect_harmonic else "background"
            if ml_unavailable and (harmonic_evidence_pct > 0.0 or has_suspect_harmonic):
                operator_label = "acoustic_harmonic_source"
            elif ml_unavailable:
                operator_label = "background"

        if (
            channel_count == 1
            and status == "drone_like"
            and ml_low
            and combined_drone_evidence_pct < 0.60
        ):
            status = "suspect" if has_suspect_harmonic else "background"

        if hf_negative and status in {"drone_like", "alert"} and not strong_multichannel_evidence:
            status = "suspect" if harmonic_evidence_pct > 0.0 or has_suspect_harmonic else "background"

        if status != "alert" and operator_label == "drone_like":
            label_allowed = (
                (
                    ml_drone_like_persistent
                    and ml_drone_pct is not None
                    and float(ml_drone_pct) >= self._effective_strong_threshold()
                    and combined_drone_evidence_pct >= 0.45
                )
                or strong_multichannel_evidence
            )
            if not label_allowed:
                if ml_drone_pct is not None and float(ml_drone_pct) >= self._effective_candidate_threshold():
                    operator_label = "ml_drone_candidate"
                elif harmonic_evidence_pct > 0.0:
                    operator_label = "background"
                else:
                    operator_label = "background"
                if status == "drone_like":
                    status = "suspect" if has_suspect_harmonic else "background"

        return status, operator_label

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
            int(self.config.min_ml_candidate_windows),
            int(self.config.min_ml_drone_like_windows),
            5,
            int(np.ceil(float(self.config.ml_strong_recent_window_sec))),
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
        channel_count: int = 1,
    ) -> float:
        confidence = _sigmoid_score(evidence_score, self.suspect_threshold)
        if has_suspect_harmonic:
            support = max(float(hf_p_drone or 0.0), float(cnn_p_drone or 0.0))
            confidence = min(1.0, confidence + 0.15 * support)
            if channel_count >= 2 and agreement_count >= 2:
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
        combined_drone_evidence_pct: float = 0.0,
        hf_negative: bool = False,
        hf_positive: bool = False,
        hf_error: bool = False,
        harmonic_activity_duration_sec: float = 0.0,
        decision_reason: str = "no acoustic evidence",
        operator_label: str = "background",
        candidate_run: int = 0,
        hf_candidate_run: int = 0,
        acoustic_candidate_run: int = 0,
        fused_candidate_run: int = 0,
        ml_positive_run: int = 0,
        strong_run: int = 0,
        hf_age_sec: Optional[float] = None,
        harmonic_age_sec: Optional[float] = None,
        max_hf_age_sec: float = 6.0,
        max_acoustic_age_sec: float = 6.0,
        estimated_detection_delay_sec: Optional[float] = None,
        raw_best_f0_hz: Optional[int] = None,
        canonical_best_f0_hz: Optional[int] = None,
        f0_family_stable: bool = False,
        decision_stage: str = "background",
        blocked_by: str = "",
        hf_watch_threshold: float = 0.50,
        hf_candidate_threshold: float = 0.70,
        hf_strong_threshold: float = 0.85,
        hf_candidate_pass: bool = False,
        hf_strong_pass: bool = False,
        harmonic_pass: bool = False,
        single_channel_mode: bool = True,
        candidate_block_reason: str = "",
        alert_block_reason: str = "",
        alert_blocked_reason: str = "",
        why_candidate_run_reset: str = "",
        harmonic_track_active: bool = False,
        tracked_f0_hz: Optional[int] = None,
        tracked_ridges: Optional[list[dict[str, float | int]]] = None,
        harmonic_track_age_sec: float = 0.0,
        f0_raw_hz: Optional[int] = None,
        f0_track_hz: Optional[int] = None,
        f0_jump_reason: str = "",
        stable_harmonic_ridge_count: int = 0,
        longest_ridge_duration_sec: float = 0.0,
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
            combined_drone_evidence_pct=float(np.clip(combined_drone_evidence_pct, 0.0, 1.0)),
            hf_negative=bool(hf_negative),
            hf_positive=bool(hf_positive),
            hf_error=bool(hf_error),
            harmonic_activity_duration_sec=float(max(0.0, harmonic_activity_duration_sec)),
            decision_reason=str(decision_reason),
            operator_label=str(operator_label),
            candidate_run=int(candidate_run),
            hf_candidate_run=int(hf_candidate_run),
            acoustic_candidate_run=int(acoustic_candidate_run),
            fused_candidate_run=int(fused_candidate_run),
            ml_positive_run=int(ml_positive_run),
            strong_run=int(strong_run),
            hf_age_sec=None if hf_age_sec is None else float(max(0.0, hf_age_sec)),
            harmonic_age_sec=None if harmonic_age_sec is None else float(max(0.0, harmonic_age_sec)),
            max_hf_age_sec=float(max_hf_age_sec),
            max_acoustic_age_sec=float(max_acoustic_age_sec),
            estimated_detection_delay_sec=None
            if estimated_detection_delay_sec is None
            else float(max(0.0, estimated_detection_delay_sec)),
            raw_best_f0_hz=raw_best_f0_hz if raw_best_f0_hz is not None else best_f0_hz,
            canonical_best_f0_hz=canonical_best_f0_hz if canonical_best_f0_hz is not None else best_f0_hz,
            f0_family_stable=bool(f0_family_stable),
            decision_stage=str(decision_stage),
            blocked_by=str(blocked_by),
            hf_watch_threshold=float(hf_watch_threshold),
            hf_candidate_threshold=float(hf_candidate_threshold),
            hf_strong_threshold=float(hf_strong_threshold),
            hf_candidate_pass=bool(hf_candidate_pass),
            hf_strong_pass=bool(hf_strong_pass),
            harmonic_pass=bool(harmonic_pass),
            single_channel_mode=bool(single_channel_mode),
            candidate_block_reason=str(candidate_block_reason),
            alert_block_reason=str(alert_block_reason),
            alert_blocked_reason=str(alert_blocked_reason),
            why_candidate_run_reset=str(why_candidate_run_reset),
            harmonic_track_active=bool(harmonic_track_active),
            tracked_f0_hz=tracked_f0_hz,
            tracked_ridges=list(tracked_ridges or []),
            harmonic_track_age_sec=float(max(0.0, harmonic_track_age_sec)),
            f0_raw_hz=f0_raw_hz,
            f0_track_hz=f0_track_hz,
            f0_jump_reason=str(f0_jump_reason),
            stable_harmonic_ridge_count=int(stable_harmonic_ridge_count),
            longest_ridge_duration_sec=float(max(0.0, longest_ridge_duration_sec)),
        )
