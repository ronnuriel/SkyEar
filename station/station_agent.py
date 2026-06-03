from __future__ import annotations
import argparse, time
import json
from pathlib import Path
from typing import Any
import numpy as np
import requests
import yaml
from urllib.parse import urlparse, urlunparse

from shared.auth import auth_headers
from shared.event_schema import AcousticEvent, EventStatus, GeoPoint, StationHeartbeat
from station.audio_capture import audio_blocks, list_input_devices, to_mono
from station.beamforming import BeamformingResult, estimate_bearing
from station.detector_state import StationDetectorState, StationDetectorStateConfig
from station.direction import estimate_azimuth
from station.hf_detector import DEFAULT_MODEL_ID, HFDetector
from station.local_monitor import (
    atomic_write_json,
    build_local_monitor_snapshot,
    history_row_from_event,
    local_monitor_paths,
    write_local_monitor_snapshot,
)
from station.raw_recorder import RawRingBufferRecorder
from station.spectrum import compute_harmonic_lines, compute_spectrogram_summary, compute_spectrum_summary

DETECTOR_VERSION = "station-detector-state-v1"


class HFErrorReporter:
    def __init__(self):
        self._printed: set[str] = set()

    def log_once(self, error_message: str | None) -> bool:
        if not error_message:
            return False
        if error_message in self._printed:
            return False
        print(f"HF error: {error_message}")
        self._printed.add(error_message)
        return True

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
        min_ml_candidate_windows=int(hf_cfg.get("min_ml_candidate_windows", 2)),
        min_ml_drone_like_windows=int(hf_cfg.get("min_ml_drone_like_windows", 3)),
        ml_spike_single_window_caps_to_candidate=bool(hf_cfg.get("ml_spike_single_window_caps_to_candidate", True)),
        ml_strong_recent_window_sec=float(hf_cfg.get("ml_strong_recent_window_sec", 3.0)),
        f0_family_tolerance_hz=float(stability_cfg.get("f0_family_tolerance_hz", 140.0)),
    )

def _station_mode(audio: np.ndarray, direction_allowed: bool, beamforming_allowed: bool = False) -> str:
    channel_count = audio.shape[1] if audio.ndim == 2 else 1
    if beamforming_allowed:
        return "synchronized_array_beamforming"
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


def server_base_url_from_events_url(events_url: str) -> str:
    parsed = urlparse(str(events_url).strip())
    if parsed.scheme not in {"http", "https"}:
        parsed = urlparse("http://" + str(events_url).strip())
    path = parsed.path.rstrip("/")
    if path.endswith("/events"):
        path = path[: -len("/events")]
    return urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "", "")).rstrip("/")


def startup_connectivity_check(server_cfg: dict[str, Any]) -> tuple[bool, str]:
    if not bool(server_cfg.get("startup_check_enabled", True)):
        return True, "startup server check disabled"
    url = str(server_cfg.get("url") or "").strip()
    if not url:
        return False, "server.url is empty"
    base_url = server_base_url_from_events_url(url)
    timeout = float(server_cfg.get("startup_check_timeout_sec", 2.0))
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if response.status_code >= 400:
        return False, f"HTTP {response.status_code}: {response.text[:200]}"
    return True, f"{base_url}/health HTTP {response.status_code}"

def _audio_device_label(audio_cfg: dict[str, Any]) -> str | None:
    if audio_cfg.get("device_name"):
        return str(audio_cfg["device_name"])
    if audio_cfg.get("device_id") is not None:
        return f"id={audio_cfg['device_id']}"
    return None


def _mic_positions(mic_cfg: dict[str, Any]) -> np.ndarray | None:
    positions = mic_cfg.get("mic_positions_m")
    if not positions:
        return None
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    return array[:, :3] if array.shape[1] >= 3 else np.pad(array, ((0, 0), (0, 1)))


def _build_hf_detector(hf_cfg: dict[str, Any]) -> HFDetector:
    return HFDetector(
        model_id=str(hf_cfg.get("model_id") or DEFAULT_MODEL_ID),
        fallback_drone_label_idx=int(hf_cfg.get("fallback_drone_label_idx", 1)),
        threshold=float(hf_cfg.get("threshold", 0.70)),
    )


def _station_auth_headers(payload: dict[str, Any], server_cfg: dict[str, Any]) -> dict[str, str]:
    return auth_headers(
        payload,
        api_token=server_cfg.get("api_token"),
        hmac_secret=server_cfg.get("hmac_secret"),
    )


def _post_payload(url: str, payload: dict[str, Any], server_cfg: dict[str, Any], timeout: float) -> requests.Response:
    headers = _station_auth_headers(payload, server_cfg)
    if server_cfg.get("hmac_secret"):
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        headers["Content-Type"] = "application/json"
        return requests.post(url, data=body, headers=headers, timeout=timeout)
    return requests.post(url, json=payload, headers=headers, timeout=timeout)


def _is_local_candidate(event: AcousticEvent) -> bool:
    label = event.operator_label or (event.metadata or {}).get("operator_label")
    return (
        int(event.candidate_run or 0) >= 2
        or str(label) in {"local_drone_candidate", "strong_local_candidate", "drone_like", "alert"}
        or str(event.status.value if hasattr(event.status, "value") else event.status) in {"drone_like", "alert"}
    )


def _write_local_monitor(
    *,
    enabled: bool,
    state_path: Path,
    history_path: Path,
    event: AcousticEvent,
    mono: np.ndarray,
    waveform_points: int,
    spectrum: dict[str, Any],
    spectrogram: dict[str, Any],
    harmonic_lines: list[dict[str, Any]],
    hf_result: Any,
    server_state: dict[str, Any],
    history_max_rows: int | None,
    updated_unix: float,
    append_history: bool = True,
) -> None:
    if not enabled:
        return
    snapshot = build_local_monitor_snapshot(
        event=event,
        mono=mono,
        waveform_points=waveform_points,
        spectrum=spectrum,
        spectrogram=spectrogram,
        harmonic_lines=harmonic_lines,
        hf_result=hf_result,
        server_state=server_state,
        updated_unix=updated_unix,
    )
    if append_history:
        write_local_monitor_snapshot(
            state_path=state_path,
            history_path=history_path,
            snapshot=snapshot,
            history_row=history_row_from_event(event),
            history_max_rows=history_max_rows,
        )
        return
    atomic_write_json(state_path, snapshot)


def run_hf_smoke_test(cfg: dict[str, Any]) -> int:
    audio_cfg = cfg["audio"]
    hf_cfg = cfg.get("hf", {})
    detector = _build_hf_detector(hf_cfg)
    print("Capturing 1.0s audio for HF smoke test...")
    audio = next(
        audio_blocks(
            device_id=audio_cfg.get("device_id"),
            sample_rate=int(audio_cfg["sample_rate"]),
            channels=int(audio_cfg["channels"]),
            window_sec=1.0,
        )
    )
    mono = to_mono(audio)
    result = detector.predict(mono, int(audio_cfg["sample_rate"]))
    print(f"model loaded: {'yes' if detector.model_loaded else 'no'}")
    if result.error:
        print(f"HF error: {result.error}")
        return 1
    print(f"p_drone: {result.p_drone}")
    print(f"label: {result.label}")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_station.yaml")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--hf-smoke-test", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        for d in list_input_devices():
            print(d)
        return

    cfg = load_yaml(args.config)
    if args.hf_smoke_test:
        raise SystemExit(run_hf_smoke_test(cfg))

    station_cfg, audio_cfg = cfg["station"], cfg["audio"]
    det_cfg, dir_cfg, server_cfg = cfg["detector"], cfg["direction"], cfg["server"]
    mic_cfg = cfg.get("mic_array", {})
    stability_cfg = cfg.get("stability", {})
    hf_cfg = cfg.get("hf", {})
    heartbeat_cfg = cfg.get("heartbeat", {})
    raw_cfg = cfg.get("raw_recording", {})
    beamforming_cfg = cfg.get("beamforming", {})
    local_monitor_cfg = cfg.get("local_monitor", {})
    coverage_radius_m = station_cfg.get("coverage_radius_m")

    detector_state = StationDetectorState(_detector_config(det_cfg, stability_cfg, hf_cfg))
    hf_detector = None
    if bool(hf_cfg.get("enabled", False)):
        hf_detector = _build_hf_detector(hf_cfg)
    hf_error_reporter = HFErrorReporter()
    hf_run_every = max(1, int(hf_cfg.get("run_every_n_windows", 2)))
    window_index = 0
    last_hf_result = None
    last_send = 0.0
    last_heartbeat = 0.0
    last_error = None
    last_heartbeat_error = None
    heartbeat_enabled = bool(heartbeat_cfg.get("enabled", True))
    heartbeat_interval = float(heartbeat_cfg.get("interval_sec", 5.0))
    heartbeat_url = str(heartbeat_cfg.get("url") or heartbeat_url_from_events_url(server_cfg["url"]))
    mic_positions = _mic_positions(mic_cfg)
    beamforming_requested = bool(beamforming_cfg.get("enabled", False))
    beamforming_allowed = (
        beamforming_requested
        and mic_cfg.get("sync_mode") == "synchronized"
        and mic_positions is not None
        and mic_positions.shape[0] == int(audio_cfg["channels"])
    )
    direction_requested = bool(dir_cfg.get("enabled"))
    direction_allowed = direction_requested and mic_cfg.get("sync_mode") == "synchronized" and not beamforming_allowed
    if beamforming_requested and not beamforming_allowed:
        print("Beamforming disabled: synchronized mic positions do not match the audio channel count.")
    if direction_requested and not direction_allowed and not beamforming_allowed:
        print("Direction disabled: this microphone profile is not synchronized.")
    raw_recorder = None
    if bool(raw_cfg.get("enabled", False)):
        raw_recorder = RawRingBufferRecorder(
            directory=raw_cfg.get("directory", "recordings/station"),
            sample_rate=int(audio_cfg["sample_rate"]),
            channels=int(audio_cfg["channels"]),
            buffer_seconds=float(raw_cfg.get("buffer_seconds", 20.0)),
            cooldown_seconds=float(raw_cfg.get("cooldown_seconds", 5.0)),
        )
    local_monitor_enabled = bool(local_monitor_cfg.get("enabled", True))
    local_state_path, local_history_path = local_monitor_paths(cfg, str(station_cfg["station_id"]))
    local_waveform_points = int(local_monitor_cfg.get("waveform_points", 1200))
    local_history_max_rows = local_monitor_cfg.get("history_max_rows")
    local_history_max_rows = None if local_history_max_rows is None else int(local_history_max_rows)

    server_ok, server_check_reason = startup_connectivity_check(server_cfg)
    if server_ok:
        print("[SERVER] connectivity check OK:", server_check_reason)
    else:
        print("[WARN] server connectivity check failed:", server_check_reason)
        print("[WARN] continuing station in local monitor mode; central posting will keep retrying.")

    print("Starting station:", station_cfg["station_id"])

    for audio in audio_blocks(
        device_id=audio_cfg.get("device_id"),
        sample_rate=int(audio_cfg["sample_rate"]),
        channels=int(audio_cfg["channels"]),
        window_sec=float(audio_cfg.get("window_sec", 1.0)),
    ):
        now = time.time()
        sample_rate = int(audio_cfg["sample_rate"])
        mono = to_mono(audio)
        if raw_recorder is not None:
            raw_recorder.append(audio)
        if hf_detector is not None and window_index % hf_run_every == 0:
            last_hf_result = hf_detector.predict(mono, sample_rate)
        hf_p_drone = last_hf_result.p_drone if last_hf_result is not None else None
        hf_error_message = last_hf_result.error if last_hf_result is not None else None
        hf_error = bool(hf_error_message)
        hf_error_reporter.log_once(hf_error_message)
        frame = detector_state.update(audio, sample_rate, now, hf_p_drone=hf_p_drone, hf_error=hf_error)
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
        beam = BeamformingResult()
        if beamforming_allowed and audio.ndim == 2 and audio.shape[1] >= 2 and mic_positions is not None:
            beam = estimate_bearing(
                audio,
                int(audio_cfg["sample_rate"]),
                mic_positions,
                method=str(beamforming_cfg.get("method", "delay_and_sum")),
                scan_step_deg=int(beamforming_cfg.get("scan_step_deg", dir_cfg.get("scan_step_deg", 5))),
                low_hz=int(beamforming_cfg.get("low_hz", det_cfg.get("f0_min", 500))),
                high_hz=int(beamforming_cfg.get("high_hz", max_freq)),
                bearing_stability_deg=float(beamforming_cfg.get("bearing_stability_deg", 15.0)),
            )
            azimuth = beam.bearing_deg
            direction_confidence = beam.beam_score
        elif direction_allowed and audio.ndim == 2 and audio.shape[1] >= 3:
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
            hf_error=frame.hf_error,
            hf_negative=frame.hf_negative,
            hf_positive=frame.hf_positive,
            harmonic_activity_duration_sec=frame.harmonic_activity_duration_sec,
            decision_reason=frame.decision_reason,
            operator_label=frame.operator_label,
            candidate_run=frame.candidate_run,
            ml_positive_run=frame.ml_positive_run,
            strong_run=frame.strong_run,
            estimated_detection_delay_sec=frame.estimated_detection_delay_sec,
            estimated_azimuth_deg=azimuth,
            direction_confidence=direction_confidence,
            beamforming_method=beam.beamforming_method if beamforming_allowed else None,
            beam_score=beam.beam_score,
            beam_snr_gain_db=beam.beam_snr_gain_db,
            bearing_stable=beam.bearing_stable if beamforming_allowed else None,
            bearing_uncertainty_deg=beam.bearing_uncertainty_deg,
            rms=frame.rms,
            peak=frame.peak,
            duration_sec=frame.duration_sec,
            calibrated=frame.calibrated,
            strongest_channel=frame.strongest_channel,
            channel_agreement_count=frame.agreement_count,
            channel_count=frame.channel_count,
            channel_evidence=[item.__dict__ for item in frame.per_channel],
            detector_version=DETECTOR_VERSION,
            station_mode=_station_mode(audio, direction_allowed, beamforming_allowed),
            metadata={
                "sample_rate": audio_cfg["sample_rate"],
                "channels": audio_cfg["channels"],
                "coverage_radius_m": None if coverage_radius_m is None else float(coverage_radius_m),
                "mic_profile": mic_cfg.get("profile"),
                "mic_sync_mode": mic_cfg.get("sync_mode"),
                "beamforming_method": beam.beamforming_method if beamforming_allowed else None,
                "beam_score": beam.beam_score,
                "beam_snr_gain_db": beam.beam_snr_gain_db,
                "bearing_stable": beam.bearing_stable if beamforming_allowed else None,
                "bearing_uncertainty_deg": beam.bearing_uncertainty_deg,
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
                "hf_error": frame.hf_error,
                "hf_negative": frame.hf_negative,
                "hf_positive": frame.hf_positive,
                "harmonic_activity_duration_sec": frame.harmonic_activity_duration_sec,
                "decision_reason": frame.decision_reason,
                "operator_label": frame.operator_label,
                "candidate_run": frame.candidate_run,
                "ml_positive_run": frame.ml_positive_run,
                "strong_run": frame.strong_run,
                "estimated_detection_delay_sec": frame.estimated_detection_delay_sec,
                "hf_label": last_hf_result.label if last_hf_result is not None else None,
                "hf_class_probs": last_hf_result.class_probs if last_hf_result is not None else {},
                "hf_error_message": hf_error_message,
                **spectrum,
                **spectrogram,
                "harmonic_lines": harmonic_lines,
            },
        )

        hf_label = last_hf_result.label if last_hf_result is not None else None
        hf_display = f"{hf_p_drone:.2f}" if hf_p_drone is not None else "None"
        hf_mode_note = " HF unavailable — harmonic-only mode, alert disabled" if hf_error else ""
        print(
            f"{time.strftime('%H:%M:%S')} {event.status:11s} "
            f"conf={event.confidence:.2f} harm={event.harmonic_score:.1f} "
            f"th={frame.suspect_threshold:.1f}/{frame.alert_threshold:.1f} "
            f"f0={event.best_f0_hz} stable={frame.f0_stable} "
            f"hf={hf_display} label={hf_label} hf_err={hf_error} "
            f"rms={event.rms:.4f} dur={event.duration_sec:.1f} "
            f"cand={frame.candidate_run} mlrun={frame.ml_positive_run} strong={frame.strong_run} "
            f"agree={event.channel_agreement_count}/{event.channel_count} "
            f"az={event.estimated_azimuth_deg} beam={event.beam_score}{hf_mode_note}"
        )
        local_server_state = {
            "events_url": server_cfg["url"],
            "heartbeat_url": heartbeat_url,
            "last_send_error": last_error,
            "last_heartbeat_error": last_heartbeat_error,
            "last_send_unix": last_send if last_send else None,
            "last_heartbeat_unix": last_heartbeat if last_heartbeat else None,
        }
        try:
            _write_local_monitor(
                enabled=local_monitor_enabled,
                state_path=local_state_path,
                history_path=local_history_path,
                event=event,
                mono=mono,
                waveform_points=local_waveform_points,
                spectrum=spectrum,
                spectrogram=spectrogram,
                harmonic_lines=harmonic_lines,
                hf_result=last_hf_result,
                server_state=local_server_state,
                history_max_rows=local_history_max_rows,
                updated_unix=now,
            )
        except Exception as e:
            print("[WARN] local monitor write failed:", e)
        if raw_recorder is not None and _is_local_candidate(event):
            saved = raw_recorder.save_candidate(
                station_id=event.station_id,
                metadata=event.model_dump(mode="json"),
                now=now,
            )
            if saved is not None:
                wav_path, json_path = saved
                print(f"[RAW] saved candidate audio: {wav_path} sidecar={json_path}")

        if now - last_send >= float(server_cfg["send_interval_sec"]):
            try:
                event_payload = event.model_dump(mode="json")
                _post_payload(server_cfg["url"], event_payload, server_cfg, timeout=1.5)
                last_error = None
            except Exception as e:
                last_error = str(e)
                print("[WARN] send failed:", e)
            last_send = now
            local_server_state["last_send_error"] = last_error
            local_server_state["last_send_unix"] = last_send
            try:
                _write_local_monitor(
                    enabled=local_monitor_enabled,
                    state_path=local_state_path,
                    history_path=local_history_path,
                    event=event,
                    mono=mono,
                    waveform_points=local_waveform_points,
                    spectrum=spectrum,
                    spectrogram=spectrogram,
                    harmonic_lines=harmonic_lines,
                    hf_result=last_hf_result,
                    server_state=local_server_state,
                history_max_rows=local_history_max_rows,
                updated_unix=time.time(),
                append_history=False,
            )
            except Exception as e:
                print("[WARN] local monitor write failed:", e)

        if heartbeat_enabled and now - last_heartbeat >= heartbeat_interval:
            heartbeat = StationHeartbeat(
                station_id=station_cfg["station_id"],
                station_name=station_cfg.get("name"),
                timestamp_unix=now,
                status="error" if (last_error or hf_error_message) else "online",
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
                last_error=last_error or hf_error_message,
                errors=[error for error in [last_error, hf_error_message] if error],
                metadata={
                    "mic_profile": mic_cfg.get("profile"),
                    "mic_sync_mode": mic_cfg.get("sync_mode"),
                    "beamforming_enabled": beamforming_allowed,
                    "beamforming_method": beamforming_cfg.get("method") if beamforming_allowed else None,
                    "window_index": window_index,
                    "hf_error_message": hf_error_message,
                },
            )
            try:
                heartbeat_payload = heartbeat.model_dump(mode="json")
                _post_payload(heartbeat_url, heartbeat_payload, server_cfg, timeout=1.5)
                last_heartbeat_error = None
            except Exception as e:
                last_heartbeat_error = str(e)
                print("[WARN] heartbeat failed:", e)
            last_heartbeat = now
            local_server_state["last_heartbeat_error"] = last_heartbeat_error
            local_server_state["last_heartbeat_unix"] = last_heartbeat
            try:
                _write_local_monitor(
                    enabled=local_monitor_enabled,
                    state_path=local_state_path,
                    history_path=local_history_path,
                    event=event,
                    mono=mono,
                    waveform_points=local_waveform_points,
                    spectrum=spectrum,
                    spectrogram=spectrogram,
                    harmonic_lines=harmonic_lines,
                    hf_result=last_hf_result,
                    server_state=local_server_state,
                    history_max_rows=local_history_max_rows,
                    updated_unix=time.time(),
                    append_history=False,
                )
            except Exception as e:
                print("[WARN] local monitor write failed:", e)

if __name__ == "__main__":
    main()
