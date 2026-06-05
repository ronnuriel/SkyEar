from __future__ import annotations
import argparse, time
import copy
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import requests
import yaml
from urllib.parse import urlparse, urlunparse

from shared.auth import auth_headers
from shared.event_schema import AcousticEvent, EventStatus, GeoPoint, StationHeartbeat
from station.array_calibration import ArrayCalibration, apply_array_calibration, load_calibration
from station.array_profiles import array_profile
from station.audio_capture import audio_blocks, list_input_devices, to_mono
from station.audio_filters import HighPassFilter
from station.beamforming import BeamformingResult, bearing_quality_from_result, estimate_bearing
from station.bearing_tracker import BearingTracker, bearing_tracker_config_from_direction
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
from station.recording_control import RecordingControlServer
from station.recording_manager import RecordingManager
from station.spectrum import compute_harmonic_lines, compute_spectrogram_summary, compute_spectrum_summary
from station.two_mic_direction import TwoMicDirectionResult, estimate_two_mic_side

DETECTOR_VERSION = "station-detector-state-v1"
DEFAULT_CONFIG_PATH = "configs/config_station.yaml"


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


def load_config_or_exit(path: str | Path, default_path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        return load_yaml(path)
    if str(path) == default_path:
        print(
            "Default config not found. Run: skyear-copy-configs ./configs or pass --config PATH",
            file=sys.stderr,
        )
    else:
        print(
            f"Config not found: {path}. Run: skyear-copy-configs ./configs or pass --config PATH",
            file=sys.stderr,
        )
    raise SystemExit(2)


def apply_mic_array_profile_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(cfg)
    mic_cfg = resolved.setdefault("mic_array", {})
    profile_name = mic_cfg.get("profile")
    if not profile_name:
        return resolved
    profile = array_profile(str(profile_name))
    if profile is None:
        raise ValueError(f"Unknown mic_array.profile: {profile_name}")

    if not mic_cfg.get("mic_positions_m") and profile.get("mic_positions_m"):
        mic_cfg["mic_positions_m"] = profile["mic_positions_m"]
    if not mic_cfg.get("sync_mode") and profile.get("sync_mode"):
        mic_cfg["sync_mode"] = profile["sync_mode"]

    beam_profile = profile.get("beamforming") or {}
    beam_cfg = resolved.setdefault("beamforming", {})
    for key in ("low_hz", "high_hz"):
        if key not in beam_cfg and key in beam_profile:
            beam_cfg[key] = beam_profile[key]
    return resolved

def _detector_config(
    det_cfg: dict[str, Any],
    stability_cfg: dict[str, Any] | None = None,
    hf_cfg: dict[str, Any] | None = None,
    detection_cfg: dict[str, Any] | None = None,
    harmonic_cfg: dict[str, Any] | None = None,
) -> StationDetectorStateConfig:
    stability_cfg = stability_cfg or {}
    hf_cfg = hf_cfg or {}
    detection_cfg = detection_cfg or {}
    harmonic_cfg = harmonic_cfg or {}
    profile = str(detection_cfg.get("profile", det_cfg.get("profile", "conservative")))
    return StationDetectorStateConfig(
        detection_profile=profile,
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
        hf_watch_threshold=float(detection_cfg.get("hf_watch_threshold", hf_cfg.get("hf_watch_threshold", 0.50))),
        hf_candidate_threshold=float(detection_cfg.get("hf_candidate_threshold", hf_cfg.get("hf_candidate_threshold", 0.70))),
        hf_strong_threshold=float(detection_cfg.get("hf_strong_threshold", hf_cfg.get("hf_strong_threshold", 0.85))),
        ml_positive_threshold=float(detection_cfg.get("ml_positive_threshold", hf_cfg.get("ml_positive_threshold", hf_cfg.get("ml_strong_threshold", 0.90)))),
        single_mic_candidate_run_required=int(detection_cfg.get("single_mic_candidate_run_required", hf_cfg.get("single_mic_candidate_run_required", 2))),
        single_mic_strong_run_required=int(detection_cfg.get("single_mic_strong_run_required", hf_cfg.get("single_mic_strong_run_required", 3))),
        allow_single_mic_alert=bool(detection_cfg.get("allow_single_mic_alert", hf_cfg.get("allow_single_mic_alert", False))),
        hf_required_for_single_channel_alert=bool(hf_cfg.get("required_for_single_channel_alert", True)),
        hf_negative_caps_status=bool(hf_cfg.get("negative_caps_status", True)),
        ml_strong_threshold=float(hf_cfg.get("ml_strong_threshold", detection_cfg.get("ml_positive_threshold", 0.90))),
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
        max_hf_age_sec=float(hf_cfg.get("max_age_sec", detection_cfg.get("max_hf_age_sec", 6.0))),
        max_acoustic_age_sec=float(detection_cfg.get("max_acoustic_age_sec", harmonic_cfg.get("max_acoustic_age_sec", 6.0))),
        harmonic_lock_enabled=bool(harmonic_cfg.get("lock_enabled", True)),
        harmonic_lock_min_duration_sec=float(harmonic_cfg.get("lock_min_duration_sec", 3.0)),
        harmonic_lock_hold_sec=float(harmonic_cfg.get("lock_hold_sec", 5.0)),
        harmonic_f0_jump_penalty=float(harmonic_cfg.get("f0_jump_penalty", 0.5)),
        harmonic_ridge_max_drift_hz=float(harmonic_cfg.get("ridge_max_drift_hz", 80.0)),
        harmonic_track_bandwidth_hz=float(harmonic_cfg.get("track_bandwidth_hz", 120.0)),
        harmonic_noise_floor_rolling_median_sec=float(harmonic_cfg.get("noise_floor_rolling_median_sec", 10.0)),
    )

def _station_mode(
    audio: np.ndarray,
    direction_allowed: bool,
    beamforming_allowed: bool = False,
    two_mic_direction_allowed: bool = False,
) -> str:
    channel_count = audio.shape[1] if audio.ndim == 2 else 1
    if beamforming_allowed:
        return "synchronized_array_beamforming"
    if direction_allowed:
        return "synchronized_array_direction"
    if two_mic_direction_allowed:
        return "two_mic_left_right"
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


def _selected_audio_device_summary(audio_cfg: dict[str, Any]) -> dict[str, Any]:
    device_id = audio_cfg.get("device_id")
    device_name = audio_cfg.get("device_name")
    try:
        devices = list_input_devices()
    except Exception:
        devices = []
    if device_name:
        selected_name = str(device_name)
    elif device_id is None:
        selected_name = "system_default"
    else:
        try:
            lookup_id: Any = int(device_id)
        except (TypeError, ValueError):
            lookup_id = device_id
        selected_name = next(
            (str(item.get("name")) for item in devices if item.get("id") == lookup_id),
            "unknown",
        )
    return {
        "selected_audio_device_id": device_id,
        "selected_audio_device_name": selected_name,
        "channels": int(audio_cfg["channels"]),
        "sample_rate": int(audio_cfg["sample_rate"]),
    }


def _build_audio_highpass_filter(audio_cfg: dict[str, Any]) -> HighPassFilter | None:
    cutoff_hz = float(audio_cfg.get("highpass_hz") or 0.0)
    if cutoff_hz <= 0.0:
        return None
    return HighPassFilter(
        sample_rate=int(audio_cfg["sample_rate"]),
        cutoff_hz=cutoff_hz,
        channels=int(audio_cfg["channels"]),
        order=int(audio_cfg.get("highpass_order", 4)),
    )


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _apply_heading_offset(bearing_deg: float | None, heading_offset_deg: float | None) -> float | None:
    if bearing_deg is None:
        return None
    return (float(bearing_deg) + float(heading_offset_deg or 0.0)) % 360.0


def _station_id(station_cfg: dict[str, Any]) -> str:
    return str(station_cfg.get("station_id") or station_cfg.get("id") or "station_001")


def _station_geo_fields(station_cfg: dict[str, Any]) -> dict[str, Any]:
    latitude = _optional_float(station_cfg.get("latitude"))
    longitude = _optional_float(station_cfg.get("longitude"))
    altitude = _optional_float(station_cfg.get("altitude_m"))
    heading_offset = float(station_cfg.get("heading_offset_deg") or 0.0)
    location_label = str(station_cfg.get("location_label") or "")
    return {
        "station_latitude": latitude,
        "station_longitude": longitude,
        "station_altitude_m": altitude,
        "station_heading_offset_deg": heading_offset,
        "station_location_label": location_label,
    }


def _station_location(station_cfg: dict[str, Any]) -> GeoPoint | None:
    fields = _station_geo_fields(station_cfg)
    if fields["station_latitude"] is None or fields["station_longitude"] is None:
        return None
    return GeoPoint(
        latitude=float(fields["station_latitude"]),
        longitude=float(fields["station_longitude"]),
        altitude_m=fields["station_altitude_m"],
    )


def _mic_positions(mic_cfg: dict[str, Any]) -> np.ndarray | None:
    positions = mic_cfg.get("mic_positions_m")
    if not positions:
        return None
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    return array[:, :3] if array.shape[1] >= 3 else np.pad(array, ((0, 0), (0, 1)))


def load_array_calibration_for_station(dir_cfg: dict[str, Any], audio_cfg: dict[str, Any]) -> ArrayCalibration | None:
    path = dir_cfg.get("calibration_file")
    channels = int(audio_cfg.get("channels", 1))
    if not path:
        if channels >= 4:
            print("Array uncalibrated; bearing may be unreliable")
        return None
    try:
        calibration = load_calibration(path)
    except Exception as exc:
        print(f"[WARN] could not load array calibration_file={path}: {type(exc).__name__}: {exc}")
        if channels >= 4:
            print("Array uncalibrated; bearing may be unreliable")
        return None
    if calibration is not None:
        if not calibration.calibration_valid:
            print(f"[WARN] array calibration is placeholder/invalid: {path}")
            if channels >= 4:
                print("Array uncalibrated; bearing may be unreliable")
        else:
            print(f"Array calibration loaded: {path}")
    return calibration


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


def _execute_recording_command(manager: RecordingManager, command: dict[str, Any] | None) -> dict[str, Any] | None:
    if not command:
        return None
    action = str(command.get("action") or "").lower()
    payload = command.get("payload") or {}
    try:
        if action == "start":
            state = manager.start_recording(
                session_name=payload.get("session_name"),
                label=payload.get("label"),
                note=payload.get("note"),
            )
        elif action == "stop":
            state = manager.stop_recording()
        elif action == "mark":
            state = manager.mark_event(
                label=str(payload.get("label") or "unknown_noise"),
                note=payload.get("note"),
                distance_m=_optional_float(payload.get("distance_m")),
                bearing_deg=_optional_float(payload.get("bearing_deg")),
                drone_model=payload.get("drone_model"),
            )
        else:
            return {"ok": False, "command_id": command.get("command_id"), "error": f"unknown_action:{action}"}
        return {"ok": True, "command_id": command.get("command_id"), "action": action, "state": state}
    except Exception as exc:
        return {"ok": False, "command_id": command.get("command_id"), "action": action, "error": f"{type(exc).__name__}: {exc}"}


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
    beam_result: BeamformingResult | None,
    server_state: dict[str, Any],
    recording_state: dict[str, Any] | None = None,
    history_max_rows: int | None = None,
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
        beam_result=beam_result,
        server_state=server_state,
        recording_state=recording_state,
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
    audio_highpass = _build_audio_highpass_filter(audio_cfg)
    print("Capturing 1.0s audio for HF smoke test...")
    audio = next(
        audio_blocks(
            device_id=audio_cfg.get("device_id"),
            sample_rate=int(audio_cfg["sample_rate"]),
            channels=int(audio_cfg["channels"]),
            window_sec=1.0,
        )
    )
    if audio_highpass is not None:
        audio = audio_highpass.process(audio)
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
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--hf-smoke-test", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        for d in list_input_devices():
            print(d)
        return

    cfg = load_config_or_exit(args.config)
    try:
        cfg = apply_mic_array_profile_defaults(cfg)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    if args.hf_smoke_test:
        raise SystemExit(run_hf_smoke_test(cfg))

    station_cfg, audio_cfg = cfg["station"], cfg["audio"]
    station_id = _station_id(station_cfg)
    station_geo = _station_geo_fields(station_cfg)
    det_cfg, dir_cfg, server_cfg = cfg["detector"], cfg["direction"], cfg["server"]
    detection_cfg = cfg.get("detection", {})
    mic_cfg = cfg.get("mic_array", {})
    stability_cfg = cfg.get("stability", {})
    hf_cfg = cfg.get("hf", {})
    harmonic_cfg = cfg.get("harmonic", {})
    heartbeat_cfg = cfg.get("heartbeat", {})
    raw_cfg = cfg.get("raw_recording", {})
    recording_cfg = cfg.get("recording", {})
    beamforming_cfg = cfg.get("beamforming", {})
    two_mic_direction_cfg = cfg.get("two_mic_direction", {})
    local_monitor_cfg = cfg.get("local_monitor", {})
    coverage_radius_m = station_cfg.get("coverage_radius_m")

    detector_state = StationDetectorState(_detector_config(det_cfg, stability_cfg, hf_cfg, detection_cfg, harmonic_cfg))
    bearing_tracker = BearingTracker(bearing_tracker_config_from_direction(dir_cfg))
    hf_detector = None
    if bool(hf_cfg.get("enabled", False)):
        hf_detector = _build_hf_detector(hf_cfg)
    hf_error_reporter = HFErrorReporter()
    hf_run_every = max(1, int(hf_cfg.get("run_every_n_windows", 2)))
    window_index = 0
    last_hf_result = None
    last_hf_at = None
    last_send = 0.0
    last_heartbeat = 0.0
    last_error = None
    last_heartbeat_error = None
    last_recording_command_result = None
    heartbeat_enabled = bool(heartbeat_cfg.get("enabled", True))
    heartbeat_interval = float(heartbeat_cfg.get("interval_sec", 5.0))
    heartbeat_url = str(heartbeat_cfg.get("url") or heartbeat_url_from_events_url(server_cfg["url"]))
    mic_positions = _mic_positions(mic_cfg)
    beamforming_requested = bool(beamforming_cfg.get("enabled", False))
    array_calibration = load_array_calibration_for_station(dir_cfg, audio_cfg)
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
    two_mic_direction_allowed = (
        bool(two_mic_direction_cfg.get("enabled", False))
        and int(audio_cfg["channels"]) >= 2
        and not beamforming_allowed
        and not direction_allowed
    )
    raw_recorder = None
    if bool(raw_cfg.get("enabled", False)):
        raw_recorder = RawRingBufferRecorder(
            directory=raw_cfg.get("directory", "recordings/station"),
            sample_rate=int(audio_cfg["sample_rate"]),
            channels=int(audio_cfg["channels"]),
            buffer_seconds=float(raw_cfg.get("buffer_seconds", 20.0)),
            cooldown_seconds=float(raw_cfg.get("cooldown_seconds", 5.0)),
        )
    recording_manager = RecordingManager(
        station_id=station_id,
        sample_rate=int(audio_cfg["sample_rate"]),
        channels=int(audio_cfg["channels"]),
        config=recording_cfg,
        station_config=cfg,
    )
    recording_control = None
    if bool(recording_cfg.get("enabled", True)) and bool(recording_cfg.get("local_control_enabled", True)):
        recording_control = RecordingControlServer(
            recording_manager,
            host=str(recording_cfg.get("local_control_host", "127.0.0.1")),
            port=int(recording_cfg.get("local_control_port", 8765)),
        )
        try:
            recording_control.start()
            print(
                "[RECORDING] local control:",
                f"http://{recording_control.host}:{recording_control.port}/recording/state",
            )
        except Exception as exc:
            print(f"[WARN] recording local control failed: {type(exc).__name__}: {exc}")
    local_monitor_enabled = bool(local_monitor_cfg.get("enabled", True))
    local_state_path, local_history_path = local_monitor_paths(cfg, station_id)
    local_waveform_points = int(local_monitor_cfg.get("waveform_points", 1200))
    local_history_max_rows = local_monitor_cfg.get("history_max_rows")
    local_history_max_rows = None if local_history_max_rows is None else int(local_history_max_rows)

    server_ok, server_check_reason = startup_connectivity_check(server_cfg)
    if server_ok:
        print("[SERVER] connectivity check OK:", server_check_reason)
    else:
        print("[WARN] server connectivity check failed:", server_check_reason)
        print("[WARN] continuing station in local monitor mode; central posting will keep retrying.")

    print("Starting station:", station_id)
    audio_device_summary = _selected_audio_device_summary(audio_cfg)
    audio_highpass = _build_audio_highpass_filter(audio_cfg)
    print(
        "Audio device:",
        f"selected_audio_device_id={audio_device_summary['selected_audio_device_id']}",
        f"selected_audio_device_name={audio_device_summary['selected_audio_device_name']}",
        f"channels={audio_device_summary['channels']}",
        f"sample_rate={audio_device_summary['sample_rate']}",
    )
    if audio_highpass is not None:
        print(
            "Audio preprocessing:",
            f"highpass_hz={audio_highpass.cutoff_hz:g}",
            f"highpass_order={audio_highpass.order}",
        )
    if two_mic_direction_allowed:
        print(
            "Two-mic direction:",
            "enabled",
            f"spacing_m={float(two_mic_direction_cfg.get('spacing_m', 0.5)):g}",
            f"left_channel={int(two_mic_direction_cfg.get('left_channel', 0)) + 1}",
            f"right_channel={int(two_mic_direction_cfg.get('right_channel', 1)) + 1}",
        )

    for audio in audio_blocks(
        device_id=audio_cfg.get("device_id"),
        sample_rate=int(audio_cfg["sample_rate"]),
        channels=int(audio_cfg["channels"]),
        window_sec=float(audio_cfg.get("window_sec", 1.0)),
    ):
        now = time.time()
        sample_rate = int(audio_cfg["sample_rate"])
        recording_manager.append_audio(audio, timestamp=now)
        if raw_recorder is not None:
            raw_recorder.append(audio)
        if audio_highpass is not None:
            audio = audio_highpass.process(audio)
        mono = to_mono(audio)
        if hf_detector is not None and window_index % hf_run_every == 0:
            last_hf_result = hf_detector.predict(mono, sample_rate)
            last_hf_at = now
        hf_age_sec = None if last_hf_at is None else max(0.0, now - float(last_hf_at))
        hf_p_drone = last_hf_result.p_drone if last_hf_result is not None else None
        hf_error_message = last_hf_result.error if last_hf_result is not None else None
        hf_error = bool(hf_error_message)
        hf_error_reporter.log_once(hf_error_message)
        frame = detector_state.update(
            audio,
            sample_rate,
            now,
            hf_p_drone=hf_p_drone,
            hf_error=hf_error,
            hf_age_sec=hf_age_sec,
        )
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
        two_mic = TwoMicDirectionResult()
        beam_audio = audio
        beam_positions = mic_positions
        calibration_meta = {
            "calibration_loaded": array_calibration is not None,
            "calibration_file": None if array_calibration is None else array_calibration.source_path,
        }
        if beamforming_allowed and audio.ndim == 2 and audio.shape[1] >= 2 and mic_positions is not None:
            try:
                beam_audio, beam_positions, calibration_meta = apply_array_calibration(
                    audio,
                    mic_positions,
                    array_calibration,
                )
            except Exception as exc:
                calibration_meta = {
                    "calibration_loaded": array_calibration is not None,
                    "calibration_file": None if array_calibration is None else array_calibration.source_path,
                    "calibration_error": f"{type(exc).__name__}: {exc}",
                }
                print("[WARN] array calibration apply failed:", calibration_meta["calibration_error"])
                beam_audio = audio
                beam_positions = mic_positions
            if array_calibration is not None and not array_calibration.calibration_valid:
                beam = BeamformingResult(
                    beamforming_method=str(beamforming_cfg.get("method", "delay_and_sum")),
                    bearing_reliable=False,
                    bearing_reject_reason="array_calibration_invalid",
                )
            else:
                beam = estimate_bearing(
                    beam_audio,
                    int(audio_cfg["sample_rate"]),
                    beam_positions,
                    method=str(beamforming_cfg.get("method", "delay_and_sum")),
                    scan_step_deg=int(beamforming_cfg.get("scan_step_deg", dir_cfg.get("scan_step_deg", 5))),
                    low_hz=int(beamforming_cfg.get("low_hz", det_cfg.get("f0_min", 500))),
                    high_hz=int(beamforming_cfg.get("high_hz", max_freq)),
                    bearing_stability_deg=float(beamforming_cfg.get("bearing_stability_deg", 15.0)),
                    min_beam_confidence_pct=float(
                        dir_cfg.get(
                            "min_beam_confidence_pct",
                            beamforming_cfg.get("min_beam_confidence_pct", 0.55),
                        )
                    ),
                    min_peak_ratio=float(dir_cfg.get("min_peak_ratio", beamforming_cfg.get("min_peak_ratio", 1.3))),
                    max_second_peak_ratio=float(
                        dir_cfg.get(
                            "max_second_peak_ratio",
                            beamforming_cfg.get("max_second_peak_ratio", 0.85),
                        )
                    ),
                    reject_ambiguous_bearing=bool(
                        dir_cfg.get(
                            "reject_ambiguous_bearing",
                            beamforming_cfg.get("reject_ambiguous_bearing", True),
                        )
                    ),
                    include_scan=True,
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
            beam.bearing_reliable = azimuth is not None
            beam.bearing_reject_reason = None if azimuth is not None else "direction_estimator_no_bearing"
        elif two_mic_direction_allowed:
            two_mic = estimate_two_mic_side(
                audio,
                int(audio_cfg["sample_rate"]),
                spacing_m=float(two_mic_direction_cfg.get("spacing_m", 0.5)),
                left_channel=int(two_mic_direction_cfg.get("left_channel", 0)),
                right_channel=int(two_mic_direction_cfg.get("right_channel", 1)),
                low_hz=int(two_mic_direction_cfg.get("low_hz", det_cfg.get("f0_min", 500))),
                high_hz=int(two_mic_direction_cfg.get("high_hz", det_cfg.get("max_freq", 6000))),
                min_delay_us=float(two_mic_direction_cfg.get("min_delay_us", 40.0)),
                min_peak_ratio=float(two_mic_direction_cfg.get("min_peak_ratio", 1.2)),
                min_rms=float(two_mic_direction_cfg.get("min_rms", 1e-5)),
            )
        raw_azimuth = _apply_heading_offset(azimuth, station_geo["station_heading_offset_deg"])
        raw_bearing_reliable = bool(getattr(beam, "bearing_reliable", False)) if beamforming_allowed else raw_azimuth is not None
        bearing_quality = bearing_quality_from_result(beam) if raw_azimuth is not None else None
        if bearing_quality is None and raw_azimuth is not None:
            bearing_quality = "fair" if raw_bearing_reliable else "unreliable"
        bearing_track = bearing_tracker.update(
            timestamp=now,
            raw_bearing_deg=raw_azimuth,
            bearing_quality=bearing_quality,
            bearing_reliable=raw_bearing_reliable,
            beam_confidence_pct=beam.beam_confidence_pct,
            peak_ratio=beam.peak_ratio,
            second_peak_ratio=beam.second_peak_ratio,
            bearing_reject_reason=beam.bearing_reject_reason,
        )
        azimuth = bearing_track.tracked_bearing_deg if bearing_track.bearing_used_for_geo else None
        bearing_reliable = bool(bearing_track.bearing_used_for_geo)
        bearing_reject_reason = bearing_track.bearing_reject_reason
        if bearing_track.bearing_flip_suppressed:
            bearing_reject_reason = bearing_reject_reason or "bearing_flip_suppressed"
            bearing_quality = "unreliable"
        bearing_uncertainty_deg = (
            bearing_track.bearing_uncertainty_deg
            if bearing_track.bearing_uncertainty_deg is not None
            else beam.bearing_uncertainty_deg
        )

        event = AcousticEvent(
            station_id=station_id,
            station_name=station_cfg.get("name"),
            timestamp_unix=now,
            station_location=_station_location(station_cfg),
            **station_geo,
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
            hf_candidate_run=frame.hf_candidate_run,
            acoustic_candidate_run=frame.acoustic_candidate_run,
            fused_candidate_run=frame.fused_candidate_run,
            ml_positive_run=frame.ml_positive_run,
            strong_run=frame.strong_run,
            hf_age_sec=frame.hf_age_sec,
            harmonic_age_sec=frame.harmonic_age_sec,
            max_hf_age_sec=frame.max_hf_age_sec,
            max_acoustic_age_sec=frame.max_acoustic_age_sec,
            estimated_detection_delay_sec=frame.estimated_detection_delay_sec,
            decision_stage=frame.decision_stage,
            blocked_by=frame.blocked_by,
            hf_watch_threshold=frame.hf_watch_threshold,
            hf_candidate_threshold=frame.hf_candidate_threshold,
            hf_strong_threshold=frame.hf_strong_threshold,
            hf_candidate_pass=frame.hf_candidate_pass,
            hf_strong_pass=frame.hf_strong_pass,
            harmonic_pass=frame.harmonic_pass,
            single_channel_mode=frame.single_channel_mode,
            candidate_block_reason=frame.candidate_block_reason,
            alert_block_reason=frame.alert_block_reason,
            alert_blocked_reason=frame.alert_blocked_reason,
            why_candidate_run_reset=frame.why_candidate_run_reset,
            harmonic_track_active=frame.harmonic_track_active,
            tracked_f0_hz=frame.tracked_f0_hz,
            tracked_ridges=frame.tracked_ridges,
            harmonic_track_age_sec=frame.harmonic_track_age_sec,
            f0_raw_hz=frame.f0_raw_hz,
            f0_track_hz=frame.f0_track_hz,
            f0_jump_reason=frame.f0_jump_reason,
            stable_harmonic_ridge_count=frame.stable_harmonic_ridge_count,
            longest_ridge_duration_sec=frame.longest_ridge_duration_sec,
            estimated_azimuth_deg=azimuth,
            raw_bearing_deg=raw_azimuth,
            tracked_bearing_deg=bearing_track.tracked_bearing_deg,
            bearing_velocity_deg_per_sec=bearing_track.bearing_velocity_deg_per_sec,
            bearing_track_age_sec=bearing_track.bearing_track_age_sec,
            bearing_track_stable=bearing_track.bearing_track_stable,
            bearing_track_status=bearing_track.bearing_track_status,
            bearing_flip_suppressed=bearing_track.bearing_flip_suppressed,
            bearing_used_for_geo=bearing_track.bearing_used_for_geo,
            direction_confidence=direction_confidence,
            beamforming_method=beam.beamforming_method if beamforming_allowed else None,
            beam_score=beam.beam_score,
            beam_snr_gain_db=beam.beam_snr_gain_db,
            beam_confidence_pct=beam.beam_confidence_pct,
            beam_peak_to_median=beam.beam_peak_to_median,
            beam_peak_to_second_peak=beam.beam_peak_to_second_peak,
            second_peak_bearing_deg=beam.second_peak_bearing_deg,
            second_peak_ratio=beam.second_peak_ratio,
            peak_ratio=beam.peak_ratio,
            bearing_ambiguity_deg=beam.bearing_ambiguity_deg,
            bearing_reliable=bearing_reliable,
            bearing_reject_reason=bearing_reject_reason,
            bearing_quality=bearing_quality,
            bearing_stable=beam.bearing_stable if beamforming_allowed else None,
            bearing_uncertainty_deg=bearing_uncertainty_deg,
            rms=frame.rms,
            peak=frame.peak,
            duration_sec=frame.duration_sec,
            calibrated=frame.calibrated,
            strongest_channel=frame.strongest_channel,
            channel_agreement_count=frame.agreement_count,
            channel_count=frame.channel_count,
            channel_evidence=[item.__dict__ for item in frame.per_channel],
            detector_version=DETECTOR_VERSION,
            station_mode=_station_mode(audio, direction_allowed, beamforming_allowed, two_mic_direction_allowed),
            metadata={
                "sample_rate": audio_cfg["sample_rate"],
                "channels": audio_cfg["channels"],
                "audio_highpass_hz": None if audio_highpass is None else audio_highpass.cutoff_hz,
                "audio_highpass_order": None if audio_highpass is None else audio_highpass.order,
                "latitude": station_geo["station_latitude"],
                "longitude": station_geo["station_longitude"],
                "altitude_m": station_geo["station_altitude_m"],
                "heading_offset_deg": station_geo["station_heading_offset_deg"],
                "location_label": station_geo["station_location_label"],
                "coverage_radius_m": None if coverage_radius_m is None else float(coverage_radius_m),
                "mic_profile": mic_cfg.get("profile"),
                "mic_sync_mode": mic_cfg.get("sync_mode"),
                **calibration_meta,
                "beamforming_method": beam.beamforming_method if beamforming_allowed else None,
                "beam_score": beam.beam_score,
                "beam_snr_gain_db": beam.beam_snr_gain_db,
                "beam_confidence_pct": beam.beam_confidence_pct,
                "beam_peak_to_median": beam.beam_peak_to_median,
                "beam_peak_to_second_peak": beam.beam_peak_to_second_peak,
                "raw_estimated_azimuth_deg": raw_azimuth,
                "second_peak_bearing_deg": beam.second_peak_bearing_deg,
                "second_peak_ratio": beam.second_peak_ratio,
                "peak_ratio": beam.peak_ratio,
                "bearing_ambiguity_deg": beam.bearing_ambiguity_deg,
                "bearing_reliable": bearing_reliable,
                "bearing_reject_reason": bearing_reject_reason,
                "bearing_quality": bearing_quality,
                "bearing_stable": beam.bearing_stable if beamforming_allowed else None,
                "bearing_uncertainty_deg": bearing_uncertainty_deg,
                "raw_bearing_deg": raw_azimuth,
                "tracked_bearing_deg": bearing_track.tracked_bearing_deg,
                "bearing_velocity_deg_per_sec": bearing_track.bearing_velocity_deg_per_sec,
                "bearing_track_age_sec": bearing_track.bearing_track_age_sec,
                "bearing_track_stable": bearing_track.bearing_track_stable,
                "bearing_track_status": bearing_track.bearing_track_status,
                "bearing_flip_suppressed": bearing_track.bearing_flip_suppressed,
                "bearing_used_for_geo": bearing_track.bearing_used_for_geo,
                "two_mic_direction_enabled": two_mic_direction_allowed,
                "two_mic_side": two_mic.side,
                "two_mic_delay_us": two_mic.delay_us,
                "two_mic_confidence": two_mic.confidence,
                "two_mic_peak_ratio": two_mic.peak_ratio,
                "two_mic_reason": two_mic.reason,
                "two_mic_spacing_m": float(two_mic_direction_cfg.get("spacing_m", 0.5))
                if two_mic_direction_allowed
                else None,
                "two_mic_left_channel": int(two_mic_direction_cfg.get("left_channel", 0)) if two_mic_direction_allowed else None,
                "two_mic_right_channel": int(two_mic_direction_cfg.get("right_channel", 1)) if two_mic_direction_allowed else None,
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
                "hf_candidate_run": frame.hf_candidate_run,
                "acoustic_candidate_run": frame.acoustic_candidate_run,
                "fused_candidate_run": frame.fused_candidate_run,
                "ml_positive_run": frame.ml_positive_run,
                "strong_run": frame.strong_run,
                "hf_age_sec": frame.hf_age_sec,
                "harmonic_age_sec": frame.harmonic_age_sec,
                "max_hf_age_sec": frame.max_hf_age_sec,
                "max_acoustic_age_sec": frame.max_acoustic_age_sec,
                "estimated_detection_delay_sec": frame.estimated_detection_delay_sec,
                "decision_stage": frame.decision_stage,
                "blocked_by": frame.blocked_by,
                "hf_watch_threshold": frame.hf_watch_threshold,
                "hf_candidate_threshold": frame.hf_candidate_threshold,
                "hf_strong_threshold": frame.hf_strong_threshold,
                "hf_candidate_pass": frame.hf_candidate_pass,
                "hf_strong_pass": frame.hf_strong_pass,
                "harmonic_pass": frame.harmonic_pass,
                "single_channel_mode": frame.single_channel_mode,
                "candidate_block_reason": frame.candidate_block_reason,
                "alert_block_reason": frame.alert_block_reason,
                "alert_blocked_reason": frame.alert_blocked_reason,
                "why_candidate_run_reset": frame.why_candidate_run_reset,
                "harmonic_track_active": frame.harmonic_track_active,
                "tracked_f0_hz": frame.tracked_f0_hz,
                "tracked_ridges": frame.tracked_ridges,
                "harmonic_track_age_sec": frame.harmonic_track_age_sec,
                "f0_raw_hz": frame.f0_raw_hz,
                "f0_track_hz": frame.f0_track_hz,
                "f0_jump_reason": frame.f0_jump_reason,
                "stable_harmonic_ridge_count": frame.stable_harmonic_ridge_count,
                "longest_ridge_duration_sec": frame.longest_ridge_duration_sec,
                "hf_label": last_hf_result.label if last_hf_result is not None else None,
                "hf_class_probs": last_hf_result.class_probs if last_hf_result is not None else {},
                "hf_error_message": hf_error_message,
                **spectrum,
                **spectrogram,
                "harmonic_lines": harmonic_lines,
            },
        )
        if bool(recording_cfg.get("auto_record_on_candidate", False)) and _is_local_candidate(event):
            rec_state = recording_manager.state()
            if not rec_state.get("recording"):
                recording_manager.start_recording(
                    session_name=f"{station_id}_auto_candidate",
                    label=str(event.operator_label or event.status.value),
                    note="auto_record_on_candidate",
                )

        hf_label = last_hf_result.label if last_hf_result is not None else None
        hf_display = f"{hf_p_drone:.2f}" if hf_p_drone is not None else "None"
        if frame.hf_strong_pass:
            hf_pass_text = "strong"
        elif frame.hf_candidate_pass:
            hf_pass_text = "candidate"
        elif hf_p_drone is not None and hf_p_drone >= frame.hf_watch_threshold:
            hf_pass_text = "watch"
        else:
            hf_pass_text = "none"
        hf_mode_note = " HF unavailable — harmonic-only mode, alert disabled" if hf_error else ""
        spatial_text = (
            "spatial=single_channel"
            if frame.single_channel_mode
            else f"agree={event.channel_agreement_count}/{event.channel_count}"
        )
        hf_age_text = "None" if frame.hf_age_sec is None else f"{frame.hf_age_sec:.1f}"
        acoustic_age_text = "None" if frame.harmonic_age_sec is None else f"{frame.harmonic_age_sec:.1f}"
        lock_text = "on" if frame.harmonic_track_active else "off"
        bearing_text = (
            f"bearing_raw={raw_azimuth} bearing_track={bearing_track.tracked_bearing_deg} "
            f"bearing_status={bearing_track.bearing_track_status} "
            f"bearing_flip={1 if bearing_track.bearing_flip_suppressed else 0} "
            f"bearing_geo={1 if bearing_track.bearing_used_for_geo else 0}"
        )
        side_text = f"side={two_mic.side}" if two_mic_direction_allowed else "side=n/a"
        if two_mic_direction_allowed and two_mic.delay_us is not None:
            side_text += f" dt={two_mic.delay_us:.0f}us"
        print(
            f"{time.strftime('%H:%M:%S')} {event.status:11s} "
            f"conf={event.confidence:.2f} harm={event.harmonic_score:.1f} "
            f"th={frame.suspect_threshold:.1f}/{frame.alert_threshold:.1f} "
            f"f0={event.best_f0_hz} stable={frame.f0_stable} "
            f"hf={hf_display} label={hf_label} hf_err={hf_error} "
            f"stage={frame.decision_stage} blocked={frame.blocked_by or '-'} "
            f"hfpass={hf_pass_text} single={1 if frame.single_channel_mode else 0} "
            f"cand_block={frame.candidate_block_reason or '-'} "
            f"alert_block={frame.alert_block_reason or '-'} "
            f"rms={event.rms:.4f} dur={event.duration_sec:.1f} "
            f"cand={frame.candidate_run} mlrun={frame.ml_positive_run} strong={frame.strong_run} "
            f"hf_run={frame.hf_candidate_run} acoustic_run={frame.acoustic_candidate_run} "
            f"fused_run={frame.fused_candidate_run} hf_age={hf_age_text} acoustic_age={acoustic_age_text} "
            f"track_f0={frame.f0_track_hz} f0_raw={frame.f0_raw_hz} lock={lock_text} "
            f"{bearing_text} "
            f"{spatial_text} "
            f"{side_text} az={event.estimated_azimuth_deg} beam={event.beam_score}{hf_mode_note}"
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
                beam_result=beam if beamforming_allowed else None,
                server_state=local_server_state,
                recording_state=recording_manager.state(),
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
                    beam_result=beam if beamforming_allowed else None,
                    server_state=local_server_state,
                    recording_state=recording_manager.state(),
                    history_max_rows=local_history_max_rows,
                    updated_unix=time.time(),
                    append_history=False,
                )
            except Exception as e:
                print("[WARN] local monitor write failed:", e)

        if heartbeat_enabled and now - last_heartbeat >= heartbeat_interval:
            heartbeat = StationHeartbeat(
                station_id=station_id,
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
                    "latitude": station_geo["station_latitude"],
                    "longitude": station_geo["station_longitude"],
                    "altitude_m": station_geo["station_altitude_m"],
                    "heading_offset_deg": station_geo["station_heading_offset_deg"],
                    "location_label": station_geo["station_location_label"],
                    "mic_profile": mic_cfg.get("profile"),
                    "mic_sync_mode": mic_cfg.get("sync_mode"),
                    "calibration_loaded": calibration_meta.get("calibration_loaded"),
                    "calibration_file": calibration_meta.get("calibration_file"),
                    "bad_channels": calibration_meta.get("bad_channels"),
                    "channel_health": calibration_meta.get("channel_health"),
                    "beamforming_enabled": beamforming_allowed,
                    "beamforming_method": beamforming_cfg.get("method") if beamforming_allowed else None,
                    "two_mic_direction_enabled": two_mic_direction_allowed,
                    "two_mic_side": two_mic.side,
                    "two_mic_delay_us": two_mic.delay_us,
                    "two_mic_confidence": two_mic.confidence,
                    "two_mic_peak_ratio": two_mic.peak_ratio,
                    "two_mic_reason": two_mic.reason,
                    "audio_highpass_hz": None if audio_highpass is None else audio_highpass.cutoff_hz,
                    "audio_highpass_order": None if audio_highpass is None else audio_highpass.order,
                    "window_index": window_index,
                    "hf_error_message": hf_error_message,
                    "recording_state": recording_manager.state(),
                    "recording_command_result": last_recording_command_result,
                },
            )
            try:
                heartbeat_payload = heartbeat.model_dump(mode="json")
                heartbeat_response = _post_payload(heartbeat_url, heartbeat_payload, server_cfg, timeout=1.5)
                try:
                    heartbeat_data = heartbeat_response.json()
                except Exception:
                    heartbeat_data = {}
                last_recording_command_result = _execute_recording_command(
                    recording_manager,
                    heartbeat_data.get("recording_command"),
                )
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
                    beam_result=beam if beamforming_allowed else None,
                    server_state=local_server_state,
                    recording_state=recording_manager.state(),
                    history_max_rows=local_history_max_rows,
                    updated_unix=time.time(),
                    append_history=False,
                )
            except Exception as e:
                print("[WARN] local monitor write failed:", e)

if __name__ == "__main__":
    main()
