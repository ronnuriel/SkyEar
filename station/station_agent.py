from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import Any
import numpy as np
import requests
import yaml

from shared.event_schema import AcousticEvent, EventStatus, GeoPoint
from station.audio_capture import audio_blocks, to_mono, list_input_devices
from station.detector import DetectorConfig, detect_drone_like
from station.direction import estimate_azimuth

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

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

    detector_config = DetectorConfig(
        f0_min=int(det_cfg["f0_min"]),
        f0_max=int(det_cfg["f0_max"]),
        max_freq=int(det_cfg["max_freq"]),
        suspect_threshold=float(det_cfg["suspect_threshold"]),
        alert_threshold=float(det_cfg["alert_threshold"]),
    )

    evidence_started = None
    last_send = 0.0
    print("Starting station:", station_cfg["station_id"])

    for audio in audio_blocks(
        device_id=audio_cfg.get("device_id"),
        sample_rate=int(audio_cfg["sample_rate"]),
        channels=int(audio_cfg["channels"]),
        window_sec=float(audio_cfg["window_sec"]),
    ):
        now = time.time()
        mono = to_mono(audio)
        result = detect_drone_like(mono, int(audio_cfg["sample_rate"]), detector_config)

        duration = 0.0
        if result.status in {"suspect", "alert"}:
            if evidence_started is None:
                evidence_started = now
            duration = now - evidence_started
        else:
            evidence_started = None

        status = EventStatus.BACKGROUND
        if result.status == "suspect":
            status = EventStatus.SUSPECT
        if result.status == "alert" and duration >= float(det_cfg["min_duration_sec"]):
            status = EventStatus.ALERT

        azimuth, direction_confidence = None, None
        if bool(dir_cfg["enabled"]) and audio.ndim == 2 and audio.shape[1] >= 3:
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
            status=status,
            confidence=result.confidence,
            harmonic_score=result.harmonic_score,
            best_f0_hz=result.best_f0_hz,
            estimated_azimuth_deg=azimuth,
            direction_confidence=direction_confidence,
            rms=rms(mono),
            peak=float(np.max(np.abs(mono))),
            duration_sec=duration,
            metadata={"sample_rate": audio_cfg["sample_rate"], "channels": audio_cfg["channels"]},
        )

        print(f"{time.strftime('%H:%M:%S')} {event.status:10s} conf={event.confidence:.2f} harm={event.harmonic_score:.1f} f0={event.best_f0_hz} az={event.estimated_azimuth_deg}")

        if now - last_send >= float(server_cfg["send_interval_sec"]):
            try:
                requests.post(server_cfg["url"], json=event.model_dump(mode="json"), timeout=1.5)
            except Exception as e:
                print("[WARN] send failed:", e)
            last_send = now

if __name__ == "__main__":
    main()
