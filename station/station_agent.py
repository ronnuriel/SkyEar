from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import Any
import numpy as np
import requests
import yaml

from shared.event_schema import AcousticEvent, EventStatus, GeoPoint, StationHeartbeat
from station.audio_capture import audio_blocks, list_input_devices, to_mono
from station.detector_state import StationDetectorState, StationDetectorStateConfig
from station.direction import estimate_azimuth
from station.hf_detector import HFDetector
from station.spectrum import compute_harmonic_lines, compute_spectrogram_summary, compute_spectrum_summary

DETECTOR_VERSION = "station-detector-state-v1"

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _detector_config(det_cfg: dict[str, Any], stability_cfg: dict[str, Any] | None = None, hf_cfg: dict[str, Any] | None = None) -> StationDetectorStateConfig:
    stability_cfg = stability_cfg or {}
    hf_cfg = hf_cfg or {}
    return StationDetectorStateConfig(
        f0_min=int(det_cfg.get("f0_min", 500)),
        f0_max=int(det_cfg.get("f0_max", 2200)),
        max_freq=int(det_cfg.get("max_freq", 7000)),
        min_harmonics=int(det_cfg.get("min_harmonics", 3)),
        min_suspect_threshold=float(det_cfg.get("min_suspect_threshold", det_cfg.get("suspect_threshold", 14.0))),
        min_alert_threshold=float(det_cfg.get("min_alert_threshold", det_cfg.get("alert_threshold", 22.0))),
        calibration_seconds=float(det_cfg.get("calibration_seconds", 8.0)),
        min_alert_duration_sec=float(det_cfg.get("min_alert_duration_sec", det_cfg.get("min_duration_sec", 3.0))),
        clear_after_sec=float(det_cfg.get("clear_after_sec", 2.5)),
        stability_enabled=bool(stability_cfg.get("enabled", True)),
        stability_history_windows=int(stability_cfg.get("history_windows", 4)),
        stability_max_f0_std_hz=float(stability_cfg.get("max_f0_std_hz", 80.0)),
        stability_min_score_windows=int(stability_cfg.get("min_score_windows", 3)),
        advisory_threshold=float(hf_cfg.get("threshold", 0.70)),
        hf_negative_threshold=float(hf_cfg.get("negative_threshold", 0.20)),
        hf_required_for_single_channel_alert=bool(hf_cfg.get("required_for_single_channel_alert", True)),
        hf_negative_caps_status=bool(hf_cfg.get("negative_caps_status", True)),
        ml_strong_threshold=float(hf_cfg.get("ml_strong_threshold", 0.90)),
        ml_candidate_harmonic_min_pct=float(hf_cfg.get("ml_candidate_harmonic_min_pct", 0.15)),
        ml_drone_like_harmonic_min_pct=float(hf_cfg.get("ml_drone_like_harmonic_min_pct", 0.45)),
        ml_only_duration_for_drone_like_sec=float(hf_cfg.get("ml_only_duration_for_drone_like_sec", 8.0)),
        smoothing_enabled=bool(stability_cfg.get("smoothing_enabled", True)),
        harmonic_smoothing_windows=int(stability_cfg.get("harmonic_smoothing_windows", 5)),
        harmonic_smoothing_method=str(stability_cfg.get("harmonic_smoothing_method", "median")),
        alert_enter_pct=float(stability_cfg.get("alert_enter_pct", 0.85)),
        alert_exit_pct=float(stability_cfg.get("alert_exit_pct", 0.55)),
        drone_like_enter_pct=float(stability_cfg.get("drone_like_enter_pct", 0.45)),
        drone_like_exit_pct=float(stability_cfg.get("drone_like_exit_pct", 0.25)),
        min_alert_windows=int(stability_cfg.get("min_alert_windows", 2)),
        min_drone_like_windows=int(stability_cfg.get("min_drone_like_windows", 2)),
        f0_family_tolerance_hz=float(stability_cfg.get("f0_family_tolerance_hz", 140.0)),
    )

def _station_mode(audio: np.ndarray, direction_allowed: bool) -> str:
    channel_count = audio.shape[1] if audio.ndim == 2 else 1
    if direction_allowed:
        return "synchronized_array_direction"
    if channel_count > 1:
        return "unsynchronized_multimic_voting"
    return "mono"

def heartbeat_url_from_events_url(events_url: str) -> str:
    events_url = str(events_url).rstrip("/")
    if events_url.endswith("/events"):
        return events_url[: -len("/events")] + "/stations/heartbeat"
    return events_url + "/stations/heartbeat"

def _audio_device_label(audio_cfg: dict[str, Any]) -> str | None:
    if audio_cfg.get("device_name"):
        return str(audio_cfg["device_name"])
    if audio_cfg.get("device_id") is not None:
        return f"id={audio_cfg['device_id']}"
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_station.yaml")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        for d in list_input_devices():
            print(d)
        return

    cfg = load_yaml(args.config)
    station_cfg, audio_cfg = cfg["station"], cfg["audio"]
    det_cfg, dir_cfg, server_cfg = cfg["detector"], cfg["direction"], cfg["server"]
    mic_cfg = cfg.get("mic_array", {})
    stability_cfg = cfg.get("stability", {})
    hf_cfg = cfg.get("hf", {})
    heartbeat_cfg = cfg.get("heartbeat", {})

    detector_state = StationDetectorState(_detector_config(det_cfg, stability_cfg, hf_cfg))
    hf_detector = None
    if bool(hf_cfg.get("enabled", False)):
        hf_detector = HFDetector(
            model_id=str(hf_cfg.get("model_id")),
            fallback_drone_label_idx=int(hf_cfg.get("fallback_drone_label_idx", 1)),
            threshold=float(hf_cfg.get("threshold", 0.70)),
        )
    hf_run_every = max(1, int(hf_cfg.get("run_every_n_windows", 2)))
    window_index = 0
    last_hf_result = None
    last_send = 0.0
    last_heartbeat = 0.0
    last_error = None
    heartbeat_enabled = bool(heartbeat_cfg.get("enabled", True))
    heartbeat_interval = float(heartbeat_cfg.get("interval_sec", 5.0))
    heartbeat_url = str(heartbeat_cfg.get("url") or heartbeat_url_from_events_url(server_cfg["url"]))
    direction_requested = bool(dir_cfg.get("enabled"))
    direction_allowed = direction_requested and mic_cfg.get("sync_mode") == "synchronized"
    if direction_requested and not direction_allowed:
        print("Direction disabled: this microphone profile is not synchronized.")

    print("Starting station:", station_cfg["station_id"])

    for audio in audio_blocks(
        device_id=audio_cfg.get("device_id"),
        sample_rate=int(audio_cfg["sample_rate"]),
        channels=int(audio_cfg["channels"]),
        window_sec=float(audio_cfg["window_sec"]),
    ):
        now = time.time()
        sample_rate = int(audio_cfg["sample_rate"])
        mono = to_mono(audio)
        if hf_detector is not None and window_index % hf_run_every == 0:
            last_hf_result = hf_detector.predict(mono, sample_rate)
        hf_p_drone = last_hf_result.p_drone if last_hf_result is not None else None
        frame = detector_state.update(audio, sample_rate, now, hf_p_drone=hf_p_drone)
        window_index += 1
        max_freq = int(det_cfg.get("max_freq", 7000))
        spectrum = compute_spectrum_summary(
            mono,
            sample_rate,
            max_freq=max_freq,
            n_points=300,
        )
        spectrogram = compute_spectrogram_summary(
            mono,
            sample_rate,
            max_freq=min(5000, max_freq),
            n_freq_bins=96,
            n_time_bins=64,
        )
        harmonic_lines = compute_harmonic_lines(
            frame.best_f0_hz,
            max_freq,
        )

        azimuth, direction_confidence = None, None
        if direction_allowed and audio.ndim == 2 and audio.shape[1] >= 3:
            azimuth, direction_confidence = estimate_azimuth(
                audio,
                int(audio_cfg["sample_rate"]),
                radius_m=float(dir_cfg["array_radius_m"]),
                step_deg=int(dir_cfg["scan_step_deg"]),
            )

        event = AcousticEvent(
            station_id=station_cfg["station_id"],
            station_name=station_cfg.get("name"),
            timestamp_unix=now,
            station_location=GeoPoint(
                latitude=float(station_cfg["latitude"]),
                longitude=float(station_cfg["longitude"]),
                altitude_m=float(station_cfg.get("altitude_m", 0.0)),
            ),
            status=EventStatus(frame.status),
            confidence=frame.confidence,
            harmonic_score=frame.harmonic_score,
            harmonic_score_smoothed=frame.harmonic_score_smoothed,
            harmonic_evidence_pct=frame.harmonic_evidence_pct,
            harmonic_evidence_pct_smoothed=frame.harmonic_evidence_pct_smoothed,
            best_f0_hz=frame.best_f0_hz,
            raw_best_f0_hz=frame.raw_best_f0_hz,
            canonical_best_f0_hz=frame.canonical_best_f0_hz,
            f0_family_stable=frame.f0_family_stable,
            ml_drone_pct=frame.ml_drone_pct,
            ml_drone_pct_smoothed=frame.ml_drone_pct_smoothed,
            combined_drone_evidence_pct=frame.combined_drone_evidence_pct,
            hf_p_drone=hf_p_drone,
            hf_negative=frame.hf_negative,
            hf_positive=frame.hf_positive,
            decision_reason=frame.decision_reason,
            operator_label=frame.operator_label,
            estimated_azimuth_deg=azimuth,
            direction_confidence=direction_confidence,
            rms=frame.rms,
            peak=frame.peak,
            duration_sec=frame.duration_sec,
            calibrated=frame.calibrated,
            strongest_channel=frame.strongest_channel,
            channel_agreement_count=frame.agreement_count,
            channel_count=frame.channel_count,
            channel_evidence=[item.__dict__ for item in frame.per_channel],
            detector_version=DETECTOR_VERSION,
            station_mode=_station_mode(audio, direction_allowed),
            metadata={
                "sample_rate": audio_cfg["sample_rate"],
                "channels": audio_cfg["channels"],
                "mic_profile": mic_cfg.get("profile"),
                "mic_sync_mode": mic_cfg.get("sync_mode"),
                "suspect_threshold": frame.suspect_threshold,
                "alert_threshold": frame.alert_threshold,
                "f0_stable": frame.f0_stable,
                "f0_family_stable": frame.f0_family_stable,
                "raw_best_f0_hz": frame.raw_best_f0_hz,
                "canonical_best_f0_hz": frame.canonical_best_f0_hz,
                "harmonic_score_smoothed": frame.harmonic_score_smoothed,
                "harmonic_evidence_pct": frame.harmonic_evidence_pct,
                "harmonic_evidence_pct_raw": frame.harmonic_evidence_pct_raw,
                "harmonic_evidence_pct_smoothed": frame.harmonic_evidence_pct_smoothed,
                "ml_drone_pct": frame.ml_drone_pct,
                "ml_drone_pct_smoothed": frame.ml_drone_pct_smoothed,
                "combined_drone_evidence_pct": frame.combined_drone_evidence_pct,
                "hf_negative": frame.hf_negative,
                "hf_positive": frame.hf_positive,
                "decision_reason": frame.decision_reason,
                "operator_label": frame.operator_label,
                "hf_label": last_hf_result.label if last_hf_result is not None else None,
                "hf_class_probs": last_hf_result.class_probs if last_hf_result is not None else {},
                "hf_error": last_hf_result.error if last_hf_result is not None else None,
                **spectrum,
                **spectrogram,
                "harmonic_lines": harmonic_lines,
            },
        )

        hf_label = last_hf_result.label if last_hf_result is not None else None
        hf_error = bool(last_hf_result and last_hf_result.error)
        hf_display = f"{hf_p_drone:.2f}" if hf_p_drone is not None else "None"
        print(
            f"{time.strftime('%H:%M:%S')} {event.status:11s} "
            f"conf={event.confidence:.2f} harm={event.harmonic_score:.1f} "
            f"th={frame.suspect_threshold:.1f}/{frame.alert_threshold:.1f} "
            f"f0={event.best_f0_hz} stable={frame.f0_stable} "
            f"hf={hf_display} label={hf_label} hf_err={hf_error} "
            f"rms={event.rms:.4f} dur={event.duration_sec:.1f} "
            f"agree={event.channel_agreement_count}/{event.channel_count} "
            f"az={event.estimated_azimuth_deg}"
        )

        if now - last_send >= float(server_cfg["send_interval_sec"]):
            try:
                requests.post(server_cfg["url"], json=event.model_dump(mode="json"), timeout=1.5)
                last_error = None
            except Exception as e:
                last_error = str(e)
                print("[WARN] send failed:", e)
            last_send = now

        if heartbeat_enabled and now - last_heartbeat >= heartbeat_interval:
            heartbeat = StationHeartbeat(
                station_id=station_cfg["station_id"],
                station_name=station_cfg.get("name"),
                timestamp_unix=now,
                status="error" if last_error else "online",
                station_location=event.station_location,
                audio_device=_audio_device_label(audio_cfg),
                sample_rate=sample_rate,
                channels=int(audio_cfg["channels"]),
                calibrated=frame.calibrated,
                detector_version=DETECTOR_VERSION,
                station_mode=event.station_mode,
                last_event_status=event.status.value if hasattr(event.status, "value") else str(event.status),
                last_harmonic_score=event.harmonic_score,
                last_hf_p_drone=event.hf_p_drone,
                last_error=last_error,
                errors=[last_error] if last_error else [],
                metadata={
                    "mic_profile": mic_cfg.get("profile"),
                    "mic_sync_mode": mic_cfg.get("sync_mode"),
                    "window_index": window_index,
                },
            )
            try:
                requests.post(heartbeat_url, json=heartbeat.model_dump(mode="json"), timeout=1.5)
            except Exception as e:
                print("[WARN] heartbeat failed:", e)
            last_heartbeat = now

if __name__ == "__main__":
    main()
