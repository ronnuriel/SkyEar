from __future__ import annotations

import numpy as np

from station.detector_state import StationDetectorState, StationDetectorStateConfig


SR = 44100


def _cfg(**overrides) -> StationDetectorStateConfig:
    values = {
        "calibration_seconds": 2.0,
        "min_alert_duration_sec": 3.0,
        "clear_after_sec": 2.5,
        "min_suspect_threshold": 14.0,
        "min_alert_threshold": 22.0,
    }
    values.update(overrides)
    return StationDetectorStateConfig(**values)


def _quiet(sec: float = 1.0, channels: int = 1) -> np.ndarray:
    audio = np.zeros(int(SR * sec), dtype=np.float32)
    if channels == 1:
        return audio
    return np.repeat(audio.reshape(-1, 1), channels, axis=1)


def _harmonic(sec: float = 1.0, channels: int = 1) -> np.ndarray:
    t = np.arange(int(SR * sec), dtype=np.float32) / SR
    mono = np.zeros_like(t)
    for k in range(1, 5):
        mono += (0.05 / k) * np.sin(2 * np.pi * 700 * k * t)
    if channels == 1:
        return mono.astype(np.float32)
    out = np.zeros((mono.size, channels), dtype=np.float32)
    out[:, 0] = mono
    out[:, 1:] = mono[:, None] * 0.8
    return out


def _harmonic_missing_fundamental(sec: float = 1.0, channels: int = 1) -> np.ndarray:
    t = np.arange(int(SR * sec), dtype=np.float32) / SR
    mono = np.zeros_like(t)
    for k in range(2, 6):
        mono += (0.05 / k) * np.sin(2 * np.pi * 700 * k * t)
    if channels == 1:
        return mono.astype(np.float32)
    out = np.zeros((mono.size, channels), dtype=np.float32)
    out[:, 0] = mono
    out[:, 1:] = mono[:, None] * 0.8
    return out


def _calibrated_state(**overrides) -> StationDetectorState:
    state = StationDetectorState(_cfg(**overrides))
    state.update(_quiet(), SR, 0.0)
    state.update(_quiet(), SR, 2.0)
    return state


def test_during_calibration_status_is_calibrating():
    state = StationDetectorState(_cfg())

    frame = state.update(_quiet(), SR, 0.0)

    assert frame.status == "calibrating"
    assert frame.calibrated is False


def test_after_calibration_quiet_audio_becomes_background():
    state = StationDetectorState(_cfg())

    state.update(_quiet(), SR, 0.0)
    frame = state.update(_quiet(), SR, 2.0)

    assert frame.status == "background"
    assert frame.calibrated is True


def test_single_channel_harmonic_only_stays_suspect_without_ml():
    state = _calibrated_state()

    first = state.update(_harmonic(), SR, 3.0)
    second = state.update(_harmonic(), SR, 5.0)
    third = state.update(_harmonic(), SR, 6.1)

    assert first.status == "suspect"
    assert second.status == "suspect"
    assert third.status == "suspect"
    assert third.operator_label == "acoustic_harmonic_source"
    assert third.harmonic_activity_duration_sec > 0.0


def test_signal_clears_to_background_after_clean_windows():
    state = _calibrated_state(clear_after_sec=2.5)

    state.update(_harmonic(), SR, 3.0)
    state.update(_harmonic(), SR, 4.5)
    state.update(_harmonic(), SR, 6.1)
    holding = state.update(_quiet(), SR, 7.0)
    cleared = state.update(_quiet(), SR, 10.0)

    assert holding.status == "suspect"
    assert cleared.status == "background"


def test_multichannel_input_returns_channel_scores_and_agreement_count():
    state = _calibrated_state()

    frame = state.update(_harmonic(channels=2), SR, 3.0)

    assert frame.channel_count == 2
    assert len(frame.per_channel) == 2
    assert frame.strongest_channel in {0, 1}
    assert frame.agreement_count == 2


def test_hf_support_alone_does_not_trigger_alert():
    state = _calibrated_state()

    frame = state.update(_quiet(), SR, 3.0, hf_p_drone=0.99, cnn_p_drone=0.99)

    assert frame.status == "suspect"
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.harmonic_score < frame.suspect_threshold
    assert frame.decision_reason == "ML strongly indicates drone; harmonic rotor evidence is weak"


def test_hf_negative_caps_single_channel_alert_harmonic_to_suspect():
    state = _calibrated_state()

    frame = state.update(_harmonic(), SR, 3.0, hf_p_drone=0.001)

    assert frame.harmonic_score >= frame.alert_threshold
    assert frame.harmonic_evidence_pct > 0.75
    assert frame.ml_drone_pct == 0.001
    assert frame.hf_negative is True
    assert frame.hf_positive is False
    assert frame.decision_reason == "harmonic source detected, but ML strongly rejects drone"
    assert frame.status == "suspect"


def test_hf_negative_caps_single_channel_even_when_f0_stable():
    state = _calibrated_state()

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.001)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.001)
    frame = state.update(_harmonic(), SR, 6.1, hf_p_drone=0.001)

    assert frame.harmonic_score >= frame.alert_threshold
    assert frame.f0_stable is True
    assert frame.hf_negative is True
    assert frame.status == "suspect"


def test_single_channel_harmonic_high_duration_with_hf_error_never_alerts():
    state = _calibrated_state()

    state.update(_harmonic(), SR, 3.0, hf_error=True)
    state.update(_harmonic(), SR, 4.5, hf_error=True)
    frame = state.update(_harmonic(), SR, 6.1, hf_error=True)

    assert frame.hf_error is True
    assert frame.channel_count == 1
    assert frame.agreement_count == 1
    assert frame.status == "suspect"
    assert frame.operator_label == "acoustic_harmonic_source"
    assert frame.decision_reason == "HF unavailable — harmonic-only mode, alert disabled"


def test_agree_one_of_one_does_not_count_as_multichannel_agreement():
    state = _calibrated_state()

    frame = state.update(_harmonic(), SR, 3.0, hf_error=True)

    assert frame.channel_count == 1
    assert frame.agreement_count == 1
    assert frame.status != "alert"


def test_alert_cannot_happen_with_candidate_run_zero():
    state = _calibrated_state()

    state.update(_harmonic(), SR, 3.0)
    state.update(_harmonic(), SR, 4.5)
    frame = state.update(_harmonic(), SR, 6.1)

    assert frame.candidate_run == 0
    assert frame.status != "alert"
    assert frame.status != "drone_like"


def test_hf_positive_with_high_harmonic_can_reach_alert_after_duration():
    state = _calibrated_state(allow_single_mic_alert=True)

    first = state.update(_harmonic(), SR, 3.0, hf_p_drone=0.95)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.95)
    third = state.update(_harmonic(), SR, 6.1, hf_p_drone=0.95)

    assert first.status in {"suspect", "drone_like"}
    assert third.hf_positive is True
    assert third.f0_stable is True
    assert third.decision_reason == "strong combined ML and harmonic rotor evidence"
    assert third.status == "alert"


def test_strong_ml_with_moderate_harmonic_is_operator_candidate_not_alert():
    state = _calibrated_state(min_alert_threshold=362.0)

    frame = state.update(_harmonic(), SR, 3.0, hf_p_drone=0.99)

    assert 0.20 <= frame.harmonic_evidence_pct <= 0.30
    assert frame.ml_drone_pct == 0.99
    assert frame.status == "suspect"
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.decision_reason == "ML strongly indicates drone; harmonic rotor evidence is weak"


def test_strong_ml_with_harmonic_support_becomes_drone_like():
    state = _calibrated_state(min_alert_threshold=160.0)

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.99)
    second = state.update(_harmonic(), SR, 4.5, hf_p_drone=0.99)
    frame = state.update(_harmonic(), SR, 6.0, hf_p_drone=0.99)

    assert second.status == "suspect"
    assert second.operator_label == "local_drone_candidate"
    assert frame.harmonic_evidence_pct >= 0.45
    assert frame.harmonic_evidence_pct_smoothed >= 0.45
    assert frame.status == "drone_like"
    assert frame.operator_label == "drone_like"
    assert frame.combined_drone_evidence_pct >= 0.60
    assert frame.decision_reason == "ML and harmonic rotor evidence strongly agree"


def test_combined_evidence_scores_strong_ml_partial_harmonic():
    state = _calibrated_state(min_alert_threshold=160.0)

    state.update(_harmonic(), SR, 3.0, hf_p_drone=1.0)
    frame = state.update(_harmonic(), SR, 4.5, hf_p_drone=1.0)

    expected = (2.0 * frame.ml_drone_pct * frame.harmonic_evidence_pct_smoothed) / (
        frame.ml_drone_pct + frame.harmonic_evidence_pct_smoothed + 1e-6
    )
    assert frame.combined_drone_evidence_pct == expected
    assert frame.combined_drone_evidence_pct >= 0.60
    assert frame.operator_label == "local_drone_candidate"
    assert frame.status == "suspect"


def test_single_ml_combined_spike_caps_to_candidate_not_drone_like():
    state = _calibrated_state(min_alert_threshold=160.0, smoothing_enabled=False)

    for idx in range(9):
        state.update(_quiet(), SR, 3.0 + idx, hf_p_drone=0.001)
    frame = state.update(_harmonic(), SR, 12.0, hf_p_drone=0.95)

    assert frame.ml_drone_pct >= 0.90
    assert frame.combined_drone_evidence_pct >= 0.60
    assert frame.status == "suspect"
    assert frame.operator_label == "ml_drone_candidate"


def test_one_high_hf_background_spike_stays_candidate_not_drone_like():
    state = _calibrated_state(smoothing_enabled=False)

    frame = state.update(_quiet(), SR, 3.0, hf_p_drone=0.99)

    assert frame.ml_drone_pct >= 0.90
    assert frame.combined_drone_evidence_pct == 0.0
    assert frame.status == "suspect"
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.candidate_run == 0
    assert frame.hf_candidate_run == 1
    assert frame.acoustic_candidate_run == 0
    assert frame.fused_candidate_run == 0
    assert frame.ml_positive_run == 1
    assert frame.strong_run == 0
    assert frame.candidate_block_reason == "acoustic_below_candidate_support"


def test_two_ml_high_windows_in_last_three_stay_ml_only_candidate_without_acoustics():
    state = _calibrated_state(smoothing_enabled=False)

    state.update(_quiet(), SR, 3.0, hf_p_drone=0.95)
    state.update(_quiet(), SR, 4.0, hf_p_drone=0.05)
    frame = state.update(_quiet(), SR, 5.0, hf_p_drone=0.96)

    assert state._ml_candidate_persistent() is True
    assert frame.status == "suspect"
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.candidate_run == 0
    assert frame.hf_candidate_run == 1
    assert frame.fused_candidate_run == 0
    assert frame.ml_positive_run == 1


def test_two_consecutive_ml_candidates_do_not_create_local_candidate_without_acoustics():
    state = _calibrated_state(smoothing_enabled=False)

    state.update(_quiet(), SR, 3.0, hf_p_drone=0.95)
    frame = state.update(_quiet(), SR, 4.0, hf_p_drone=0.96)

    assert frame.status == "suspect"
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.candidate_run == 0
    assert frame.hf_candidate_run == 2
    assert frame.acoustic_candidate_run == 0
    assert frame.fused_candidate_run == 0
    assert frame.ml_positive_run == 2
    assert frame.estimated_detection_delay_sec is None


def test_three_consecutive_candidates_without_harmonic_support_stay_ml_only():
    state = _calibrated_state(smoothing_enabled=False)

    state.update(_quiet(), SR, 3.0, hf_p_drone=0.95)
    state.update(_quiet(), SR, 4.0, hf_p_drone=0.96)
    frame = state.update(_quiet(), SR, 5.0, hf_p_drone=0.97)

    assert frame.status == "suspect"
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.candidate_run == 0
    assert frame.hf_candidate_run == 3
    assert frame.acoustic_candidate_run == 0
    assert frame.fused_candidate_run == 0
    assert frame.ml_positive_run == 3
    assert frame.strong_run == 0
    assert frame.combined_drone_evidence_pct == 0.0
    assert frame.why_candidate_run_reset == "acoustic_below_candidate_support"


def test_three_consecutive_ml_combined_windows_become_drone_like():
    state = _calibrated_state(min_alert_threshold=160.0, smoothing_enabled=False)

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.99)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.99)
    frame = state.update(_harmonic(), SR, 6.0, hf_p_drone=0.99)

    assert frame.ml_drone_pct >= 0.90
    assert frame.combined_drone_evidence_pct >= 0.60
    assert frame.f0_family_stable is True
    assert frame.status == "drone_like"
    assert frame.operator_label == "drone_like"
    assert frame.candidate_run == 3
    assert frame.strong_run == 3


def test_three_ml_high_windows_in_last_five_with_support_become_drone_like():
    state = _calibrated_state(min_alert_threshold=160.0, smoothing_enabled=False)

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.99)
    state.update(_quiet(), SR, 4.0, hf_p_drone=0.01)
    state.update(_harmonic(), SR, 5.0, hf_p_drone=0.98)
    state.update(_quiet(), SR, 6.0, hf_p_drone=0.01)
    frame = state.update(_harmonic(), SR, 7.0, hf_p_drone=0.97)

    assert sum(1 for value in state.ml_strong_history[-5:] if value) == 3
    assert frame.combined_drone_evidence_pct >= 0.35
    assert frame.status == "drone_like"
    assert frame.operator_label == "drone_like"


def test_background_and_helicopter_like_single_spikes_do_not_create_drone_like():
    background_state = _calibrated_state(smoothing_enabled=False)
    background = background_state.update(_quiet(), SR, 3.0, hf_p_drone=0.99)

    helicopter_state = _calibrated_state(min_alert_threshold=160.0, smoothing_enabled=False)
    helicopter_like = helicopter_state.update(_harmonic(), SR, 3.0, hf_p_drone=0.99)

    assert background.status == "suspect"
    assert background.operator_label == "ml_drone_candidate"
    assert helicopter_like.status == "suspect"
    assert helicopter_like.operator_label == "ml_drone_candidate"


def test_hf_missing_single_channel_keeps_existing_harmonic_behavior():
    state = _calibrated_state()

    state.update(_harmonic(), SR, 3.0)
    state.update(_harmonic(), SR, 4.5)
    frame = state.update(_harmonic(), SR, 6.1)

    assert frame.harmonic_score >= frame.alert_threshold
    assert frame.f0_stable is True
    assert frame.hf_negative is False
    assert frame.hf_positive is False
    assert frame.status == "suspect"
    assert frame.operator_label == "acoustic_harmonic_source"


def test_multichannel_agreement_and_stable_f0_without_hf_is_acoustic_only():
    state = _calibrated_state()

    state.update(_harmonic(channels=4), SR, 3.0)
    state.update(_harmonic(channels=4), SR, 4.5)
    frame = state.update(_harmonic(channels=4), SR, 6.1)

    assert frame.agreement_count >= 2
    assert frame.f0_stable is True
    assert frame.status == "suspect"
    assert frame.operator_label == "acoustic_harmonic_source"
    assert frame.fused_candidate_run == 0


def test_two_channel_harmonic_only_does_not_use_multichannel_alert_path():
    state = _calibrated_state()

    state.update(_harmonic(channels=2), SR, 3.0)
    state.update(_harmonic(channels=2), SR, 4.5)
    frame = state.update(_harmonic(channels=2), SR, 6.1)

    assert frame.channel_count == 2
    assert frame.agreement_count >= 2
    assert frame.status != "alert"


def test_multichannel_agreement_and_stable_f0_negative_hf_does_not_alert():
    state = _calibrated_state()

    state.update(_harmonic(channels=4), SR, 3.0, hf_p_drone=0.001)
    state.update(_harmonic(channels=4), SR, 4.5, hf_p_drone=0.001)
    frame = state.update(_harmonic(channels=4), SR, 6.1, hf_p_drone=0.001)

    assert frame.hf_negative is True
    assert frame.agreement_count >= 2
    assert frame.f0_stable is True
    assert frame.status == "suspect"
    assert frame.operator_label in {"acoustic_harmonic_source", "non_drone_harmonic"}
    assert frame.fused_candidate_run == 0


def test_alert_hysteresis_holds_briefly_then_clears():
    state = _calibrated_state(clear_after_sec=2.5, allow_single_mic_alert=True)

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.95)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.95)
    alert = state.update(_harmonic(), SR, 6.1, hf_p_drone=0.95)
    holding = state.update(_quiet(), SR, 7.0, hf_p_drone=0.95)
    cleared = state.update(_quiet(), SR, 10.0)

    assert alert.status == "alert"
    assert holding.status == "suspect"
    assert holding.fused_candidate_run == 0
    assert cleared.status == "background"


def test_f0_family_stability_treats_nearby_values_as_one_family():
    state = _calibrated_state()

    canonical_values = []
    for raw in [520, 535, 620, 635]:
        canonical = state._canonical_f0_family(raw)
        state._record_window_history(30.0, 0.50, None, raw, canonical)
        canonical_values.append(canonical)

    assert canonical_values == [520, 520, 520, 520]
    assert state._f0_family_is_stable() is True


def test_f0_family_stability_treats_octaves_as_one_family():
    state = _calibrated_state()

    first = state._canonical_f0_family(520)
    state._record_window_history(30.0, 0.50, None, 520, first)
    second = state._canonical_f0_family(1040)
    state._record_window_history(30.0, 0.50, None, 1040, second)

    assert first == 520
    assert second == 520


def test_background_label_forces_status_not_above_suspect():
    state = _calibrated_state()

    status, label = state._enforce_status_label_consistency(
        status="drone_like",
        operator_label="background",
        has_suspect_harmonic=True,
        harmonic_evidence_pct=0.2,
        ml_drone_pct=0.1,
        combined_drone_evidence_pct=0.1,
        hf_p_drone=None,
        channel_count=1,
        agreement_count=1,
        f0_family_stable=False,
    )

    assert status == "suspect"
    assert label == "background"


def test_non_drone_harmonic_label_forces_status_not_above_suspect():
    state = _calibrated_state()

    status, label = state._enforce_status_label_consistency(
        status="alert",
        operator_label="non_drone_harmonic",
        has_suspect_harmonic=True,
        harmonic_evidence_pct=0.9,
        ml_drone_pct=0.01,
        combined_drone_evidence_pct=0.01,
        hf_p_drone=0.01,
        channel_count=1,
        agreement_count=1,
        f0_family_stable=True,
    )

    assert status == "suspect"
    assert label == "non_drone_harmonic"


def test_background_like_single_channel_combined_low_cannot_be_drone_like():
    state = _calibrated_state()

    status, label = state._enforce_status_label_consistency(
        status="drone_like",
        operator_label="drone_like",
        has_suspect_harmonic=True,
        harmonic_evidence_pct=0.2,
        ml_drone_pct=0.2,
        combined_drone_evidence_pct=0.2,
        hf_p_drone=0.2,
        channel_count=1,
        agreement_count=1,
        f0_family_stable=True,
    )

    assert status == "suspect"
    assert label != "drone_like"


def test_field_debug_hf_078_stable_harmonic_two_windows_is_local_candidate_not_alert():
    state = _calibrated_state(
        detection_profile="field_debug",
        hf_candidate_threshold=0.70,
        hf_strong_threshold=0.85,
        single_mic_candidate_run_required=2,
        single_mic_strong_run_required=3,
        allow_single_mic_alert=False,
    )

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.78)
    frame = state.update(_harmonic(), SR, 4.5, hf_p_drone=0.78)

    assert frame.status == "suspect"
    assert frame.operator_label == "local_drone_candidate"
    assert frame.candidate_run == 2
    assert frame.ml_positive_run == 2
    assert frame.strong_run == 0
    assert frame.hf_candidate_pass is True
    assert frame.hf_strong_pass is False
    assert frame.harmonic_pass is True
    assert frame.single_channel_mode is True
    assert frame.alert_blocked_reason == "single_mic_alert_disabled"


def test_field_debug_hf_082_stable_harmonic_three_windows_is_strong_candidate_not_alert():
    state = _calibrated_state(
        detection_profile="field_debug",
        hf_candidate_threshold=0.70,
        hf_strong_threshold=0.80,
        single_mic_candidate_run_required=2,
        single_mic_strong_run_required=3,
        allow_single_mic_alert=False,
    )

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.82)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.82)
    frame = state.update(_harmonic(), SR, 6.0, hf_p_drone=0.82)

    assert frame.status == "suspect"
    assert frame.operator_label == "strong_local_candidate"
    assert frame.candidate_run == 3
    assert frame.ml_positive_run == 3
    assert frame.strong_run == 3
    assert frame.hf_candidate_pass is True
    assert frame.hf_strong_pass is True
    assert frame.alert_blocked_reason == "single_mic_alert_disabled"


def test_field_debug_hf_low_high_harmonic_is_harmonic_source_not_candidate():
    state = _calibrated_state(detection_profile="field_debug", allow_single_mic_alert=False)

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.10)
    frame = state.update(_harmonic(), SR, 4.5, hf_p_drone=0.10)

    assert frame.status == "suspect"
    assert frame.operator_label in {"acoustic_harmonic_source", "non_drone_harmonic"}
    assert frame.candidate_run == 0
    assert frame.hf_candidate_pass is False
    assert frame.hf_negative is True


def test_field_debug_harmonic_high_alone_never_alerts_single_channel():
    state = _calibrated_state(detection_profile="field_debug", allow_single_mic_alert=False)

    state.update(_harmonic(), SR, 3.0)
    state.update(_harmonic(), SR, 4.5)
    frame = state.update(_harmonic(), SR, 6.0)

    assert frame.channel_count == 1
    assert frame.agreement_count == 1
    assert frame.status == "suspect"
    assert frame.status != "alert"
    assert frame.operator_label == "acoustic_harmonic_source"
    assert frame.alert_blocked_reason == "single_mic_alert_disabled"


def test_conservative_profile_keeps_hf_078_below_candidate_threshold():
    state = _calibrated_state(detection_profile="conservative")

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.78)
    frame = state.update(_harmonic(), SR, 4.5, hf_p_drone=0.78)

    assert frame.status == "suspect"
    assert frame.operator_label not in {"local_drone_candidate", "strong_local_candidate"}
    assert frame.candidate_run == 0
    assert frame.hf_candidate_pass is False
    assert frame.hf_candidate_threshold == 0.90


def test_field_debug_candidate_run_cannot_emit_background_without_harmonic_support():
    state = _calibrated_state(detection_profile="field_debug", allow_single_mic_alert=False)

    frame = state.update(_quiet(), SR, 3.0, hf_p_drone=0.78)

    assert frame.candidate_run == 0
    assert frame.hf_candidate_run == 1
    assert frame.acoustic_candidate_run == 0
    assert frame.fused_candidate_run == 0
    assert frame.status == "suspect"
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.candidate_block_reason == "harmonic_below_candidate_support"
    assert frame.why_candidate_run_reset == "harmonic_below_candidate_support"


def test_field_debug_local_candidate_cannot_emit_background():
    state = _calibrated_state(detection_profile="field_debug", allow_single_mic_alert=False)

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.78)
    frame = state.update(_harmonic(), SR, 4.5, hf_p_drone=0.78)

    assert frame.operator_label == "local_drone_candidate"
    assert frame.decision_stage == "local_drone_candidate"
    assert frame.status == "suspect"


def test_field_debug_hf_drop_resets_stale_local_candidate_stage():
    state = _calibrated_state(detection_profile="field_debug", allow_single_mic_alert=False)

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.78)
    local = state.update(_harmonic(), SR, 4.5, hf_p_drone=0.78)
    dropped = state.update(_harmonic(), SR, 6.0, hf_p_drone=0.10)

    assert local.operator_label == "local_drone_candidate"
    assert dropped.candidate_run == 0
    assert dropped.hf_candidate_pass is False
    assert dropped.decision_stage not in {
        "weak_local_candidate",
        "local_drone_candidate",
        "strong_local_candidate",
        "acoustic_drone_watch",
    }
    assert dropped.operator_label in {"acoustic_harmonic_source", "non_drone_harmonic"}
    assert dropped.why_candidate_run_reset == "hf_below_candidate_threshold"


def test_harmonic_below_candidate_support_is_not_reported_as_run_reset_while_incrementing():
    state = _calibrated_state(detection_profile="field_debug", allow_single_mic_alert=False)

    frame = state.update(_quiet(), SR, 3.0, hf_p_drone=0.78)

    assert frame.candidate_run == 0
    assert frame.hf_candidate_run == 1
    assert frame.fused_candidate_run == 0
    assert frame.candidate_block_reason == "harmonic_below_candidate_support"
    assert frame.why_candidate_run_reset == "harmonic_below_candidate_support"


def test_ml_high_harmonic_zero_combined_zero_is_ml_candidate_only():
    state = _calibrated_state(smoothing_enabled=False, allow_single_mic_alert=True)

    frame = state.update(_quiet(), SR, 3.0, hf_p_drone=0.95)

    assert frame.ml_drone_pct == 0.95
    assert frame.harmonic_evidence_pct_smoothed == 0.0
    assert frame.combined_drone_evidence_pct == 0.0
    assert frame.hf_candidate_run == 1
    assert frame.acoustic_candidate_run == 0
    assert frame.fused_candidate_run == 0
    assert frame.candidate_run == 0
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.decision_stage == "ml_drone_candidate"
    assert frame.status != "alert"
    assert frame.status != "drone_like"


def test_ml_high_plus_stable_harmonic_becomes_local_candidate_after_persistence():
    state = _calibrated_state(
        detection_profile="field_debug",
        allow_single_mic_alert=False,
        smoothing_enabled=False,
        min_alert_threshold=160.0,
    )

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.95)
    frame = state.update(_harmonic(), SR, 4.5, hf_p_drone=0.95)

    assert frame.hf_candidate_run == 2
    assert frame.acoustic_candidate_run == 2
    assert frame.fused_candidate_run == 2
    assert frame.candidate_run == 2
    assert frame.operator_label == "local_drone_candidate"
    assert frame.decision_stage == "local_drone_candidate"


def test_stale_ml_high_with_low_harmonic_resets_fused_candidate():
    state = _calibrated_state(
        smoothing_enabled=False,
        allow_single_mic_alert=True,
        min_alert_threshold=160.0,
        max_hf_age_sec=6.0,
        max_acoustic_age_sec=6.0,
    )

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.95)
    local = state.update(_harmonic(), SR, 4.5, hf_p_drone=0.95)
    frame = state.update(_quiet(), SR, 12.0, hf_p_drone=0.95, hf_age_sec=7.0)

    assert local.fused_candidate_run >= 2
    assert frame.hf_age_sec is not None and frame.hf_age_sec > frame.max_hf_age_sec
    assert frame.harmonic_age_sec is not None and frame.harmonic_age_sec > frame.max_acoustic_age_sec
    assert frame.hf_candidate_pass is False
    assert frame.harmonic_pass is False
    assert frame.fused_candidate_run == 0
    assert frame.candidate_run == 0
    assert frame.status != "alert"


def test_harmonic_high_ml_low_is_non_drone_harmonic_without_alert():
    state = _calibrated_state(allow_single_mic_alert=True)

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.05)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.05)
    frame = state.update(_harmonic(), SR, 6.1, hf_p_drone=0.05)

    assert frame.harmonic_evidence_pct > 0.0
    assert frame.hf_negative is True
    assert frame.fused_candidate_run == 0
    assert frame.operator_label in {"acoustic_harmonic_source", "non_drone_harmonic"}
    assert frame.status != "alert"


def test_alert_requires_fused_evidence_not_ml_only():
    state = _calibrated_state(smoothing_enabled=False, allow_single_mic_alert=True)

    for idx in range(5):
        frame = state.update(_quiet(), SR, 3.0 + idx, hf_p_drone=0.99)

    assert frame.hf_candidate_run >= 5
    assert frame.acoustic_candidate_run == 0
    assert frame.fused_candidate_run == 0
    assert frame.combined_drone_evidence_pct == 0.0
    assert frame.operator_label == "ml_drone_candidate"
    assert frame.status != "alert"
    assert frame.status != "drone_like"


def test_harmonic_track_locks_on_stable_ridges():
    state = _calibrated_state(
        harmonic_lock_enabled=True,
        harmonic_lock_min_duration_sec=2.0,
        min_alert_threshold=160.0,
    )

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.95)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.95)
    frame = state.update(_harmonic(), SR, 6.1, hf_p_drone=0.95)

    assert frame.harmonic_track_active is True
    assert frame.f0_raw_hz is not None
    assert frame.f0_track_hz is not None
    assert abs(frame.f0_track_hz - 700) <= 80
    assert frame.harmonic_track_age_sec >= 2.0
    assert frame.stable_harmonic_ridge_count >= 3
    assert frame.longest_ridge_duration_sec >= frame.harmonic_track_age_sec


def test_harmonic_track_holds_when_fundamental_is_missing_but_ridges_continue():
    state = _calibrated_state(
        harmonic_lock_enabled=True,
        harmonic_lock_min_duration_sec=2.0,
        harmonic_lock_hold_sec=5.0,
        min_alert_threshold=160.0,
    )

    state.update(_harmonic(), SR, 3.0, hf_p_drone=0.95)
    state.update(_harmonic(), SR, 4.5, hf_p_drone=0.95)
    state.update(_harmonic(), SR, 6.1, hf_p_drone=0.95)
    frame = state.update(_harmonic_missing_fundamental(), SR, 7.2, hf_p_drone=0.95)

    assert frame.harmonic_track_active is True
    assert frame.f0_track_hz is not None
    assert abs(frame.f0_track_hz - 700) <= 80
    assert frame.stable_harmonic_ridge_count >= 3
