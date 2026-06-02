from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import Any
import numpy as np
import requests
import yaml

from shared.event_schema import AcousticEvent, EventStatus, GeoPoint
from station.audio_capture import audio_blocks, list_input_devices, to_mono
from station.detector_state import StationDetectorState, StationDetectorStateConfig
from station.direction import estimate_azimuth
from station.spectrum import compute_harmonic_lines, compute_spectrum_summary

DETECTOR_VERSION = "station-detector-state-v1"

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _detector_config(det_cfg: dict[str, Any]) -> StationDetectorStateConfig:
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
    )

def _station_mode(audio: np.ndarray, direction_allowed: bool) -> str:
    channel_count = audio.shape[1] if audio.ndim == 2 else 1
    if direction_allowed:
        return "synchronized_array_direction"
    if channel_count > 1:
        return "unsynchronized_multimic_voting"
    return "mono"

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

    detector_state = StationDetectorState(_detector_config(det_cfg))
    last_send = 0.0
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
        frame = detector_state.update(audio, int(audio_cfg["sample_rate"]), now)
        mono = to_mono(audio)
        spectrum = compute_spectrum_summary(
            mono,
            int(audio_cfg["sample_rate"]),
            max_freq=int(det_cfg.get("max_freq", 7000)),
        )
        harmonic_lines = compute_harmonic_lines(
            frame.best_f0_hz,
            int(spectrum["spectrum_max_freq_hz"]),
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
            best_f0_hz=frame.best_f0_hz,
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
                **spectrum,
                "harmonic_lines": harmonic_lines,
            },
        )

        print(
            f"{time.strftime('%H:%M:%S')} {event.status:11s} "
            f"conf={event.confidence:.2f} harm={event.harmonic_score:.1f} "
            f"f0={event.best_f0_hz} rms={event.rms:.4f} dur={event.duration_sec:.1f} "
            f"agree={event.channel_agreement_count} strong={event.strongest_channel} "
            f"az={event.estimated_azimuth_deg}"
        )

        if now - last_send >= float(server_cfg["send_interval_sec"]):
            try:
                requests.post(server_cfg["url"], json=event.model_dump(mode="json"), timeout=1.5)
            except Exception as e:
                print("[WARN] send failed:", e)
            last_send = now

if __name__ == "__main__":
    main()
