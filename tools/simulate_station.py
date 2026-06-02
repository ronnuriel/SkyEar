from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import requests

from shared.event_schema import AcousticEvent, EventStatus, GeoPoint, StationHeartbeat
from station.audio_capture import to_mono
from station.detector_state import StationDetectorState, StationDetectorStateConfig
from station.station_agent import heartbeat_url_from_events_url
from station.spectrum import compute_harmonic_lines, compute_spectrogram_summary, compute_spectrum_summary


DETECTOR_VERSION = "simulated-station-v1"


@dataclass
class SimulatedStation:
    station_id: str
    detector_state: StationDetectorState
    station_index: int = 0
    strength: float = 1.0
    delay_sec: float = 0.0


def _background_noise(rng: np.random.Generator, samples: int, channels: int, scale: float = 0.003) -> np.ndarray:
    return rng.normal(0.0, scale, size=(samples, channels)).astype(np.float32)


def _harmonic_stack(t: np.ndarray, f0_values: Iterable[float], amplitude: float, rng: np.random.Generator) -> np.ndarray:
    signal = np.zeros_like(t, dtype=np.float32)
    for f0 in f0_values:
        drift = rng.normal(0.0, 3.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        for k in range(1, 6):
            signal += (amplitude / k) * np.sin(2 * np.pi * (f0 + drift) * k * t + phase)
    return signal.astype(np.float32)


def _apply_channels(mono: np.ndarray, channels: int, rng: np.random.Generator) -> np.ndarray:
    gains = np.linspace(1.0, 0.72, channels, dtype=np.float32)
    gains *= rng.uniform(0.88, 1.08, size=channels).astype(np.float32)
    out = np.zeros((mono.size, channels), dtype=np.float32)
    for idx in range(channels):
        delay = idx % 4
        shifted = np.roll(mono, delay)
        if delay:
            shifted[:delay] = 0.0
        out[:, idx] = shifted * gains[idx]
    return out


def _drone_pass_envelope(elapsed_sec: float) -> float:
    if elapsed_sec < 6.0:
        return 0.0
    if elapsed_sec < 16.0:
        return (elapsed_sec - 6.0) / 10.0
    if elapsed_sec < 28.0:
        return 1.0
    if elapsed_sec < 36.0:
        return max(0.0, 1.0 - (elapsed_sec - 28.0) / 8.0)
    return 0.0


def generate_synthetic_audio(
    scenario: str,
    elapsed_sec: float,
    sample_rate: int = 44100,
    window_sec: float = 2.0,
    channels: int = 8,
    station_index: int = 0,
    strength: float = 1.0,
) -> np.ndarray:
    samples = int(sample_rate * window_sec)
    seed = int(elapsed_sec * 10) + station_index * 1009 + channels * 17
    rng = np.random.default_rng(seed)
    audio = _background_noise(rng, samples, channels)
    t = (np.arange(samples, dtype=np.float32) / sample_rate) + np.float32(elapsed_sec)

    scenario = scenario.lower()
    amplitude = 0.035 * strength
    if scenario == "background":
        return audio
    if scenario == "drone_hover":
        mono = _harmonic_stack(t, [1050.0 + 35.0 * station_index], amplitude, rng)
    elif scenario == "drone_pass":
        envelope = _drone_pass_envelope(elapsed_sec)
        mono = _harmonic_stack(t, [980.0 + 55.0 * station_index], amplitude * envelope, rng)
    elif scenario == "multi_rotor_jitter":
        f0_values = [1050.0, 1090.0, 1130.0, 1170.0]
        wobble = 1.0 + 0.035 * np.sin(2 * np.pi * 0.45 * elapsed_sec)
        mono = _harmonic_stack(t, f0_values, amplitude * 0.55 * wobble, rng)
    elif scenario == "false_positive_fan":
        mono = 0.018 * strength * np.sin(2 * np.pi * 170.0 * t)
        mono += 0.008 * strength * np.sin(2 * np.pi * 340.0 * t)
        mono = mono.astype(np.float32)
    elif scenario == "motorcycle_like":
        mono = rng.normal(0.0, 0.018 * strength, size=samples).astype(np.float32)
        mono += 0.018 * strength * np.sin(2 * np.pi * 140.0 * t).astype(np.float32)
        mono += 0.012 * strength * np.sin(2 * np.pi * 280.0 * t).astype(np.float32)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    audio += _apply_channels(mono, channels, rng)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def build_event(
    station: SimulatedStation,
    audio: np.ndarray,
    sample_rate: int,
    timestamp: float,
    max_freq: int = 7000,
    metadata_extra: dict | None = None,
) -> AcousticEvent:
    frame = station.detector_state.update(audio, sample_rate, timestamp)
    mono = to_mono(audio)
    spectrum = compute_spectrum_summary(mono, sample_rate, max_freq=max_freq, n_points=300)
    spectrogram = compute_spectrogram_summary(
        mono,
        sample_rate,
        max_freq=min(5000, max_freq),
        n_freq_bins=96,
        n_time_bins=64,
    )
    harmonic_lines = compute_harmonic_lines(frame.best_f0_hz, max_freq)

    metadata_extra = metadata_extra or {}
    return AcousticEvent(
        station_id=station.station_id,
        station_name=f"Simulated Station {station.station_index + 1}",
        timestamp_unix=timestamp,
        station_location=GeoPoint(latitude=32.0853, longitude=34.7818 + station.station_index * 0.002),
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
        hf_negative=frame.hf_negative,
        hf_positive=frame.hf_positive,
        decision_reason=frame.decision_reason,
        operator_label=frame.operator_label,
        candidate_run=frame.candidate_run,
        ml_positive_run=frame.ml_positive_run,
        strong_run=frame.strong_run,
        estimated_detection_delay_sec=frame.estimated_detection_delay_sec,
        rms=frame.rms,
        peak=frame.peak,
        duration_sec=frame.duration_sec,
        calibrated=frame.calibrated,
        strongest_channel=frame.strongest_channel,
        channel_agreement_count=frame.agreement_count,
        channel_count=frame.channel_count,
        channel_evidence=[item.__dict__ for item in frame.per_channel],
        detector_version=DETECTOR_VERSION,
        station_mode="unsynchronized_multimic_voting" if audio.shape[1] > 1 else "mono",
        metadata={
            "sample_rate": sample_rate,
            "channels": audio.shape[1],
            "mic_profile": "simulated_8_channel",
            "mic_sync_mode": "unsynchronized",
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
            "candidate_run": frame.candidate_run,
            "ml_positive_run": frame.ml_positive_run,
            "strong_run": frame.strong_run,
            "estimated_detection_delay_sec": frame.estimated_detection_delay_sec,
            **spectrum,
            **spectrogram,
            "harmonic_lines": harmonic_lines,
            **metadata_extra,
        },
    )


def _scenario_metadata(args: argparse.Namespace) -> dict:
    metadata = {
        "scenario_id": args.scenario_id or args.scenario,
        "simulated_source_id": args.simulated_source_id or f"{args.scenario}_source_001",
    }
    if args.coverage_radius_m is not None:
        metadata["coverage_radius_m"] = float(args.coverage_radius_m)
    if args.true_source_latitude is not None:
        metadata["true_source_latitude"] = float(args.true_source_latitude)
    if args.true_source_longitude is not None:
        metadata["true_source_longitude"] = float(args.true_source_longitude)
    if args.true_source_distance_m is not None:
        metadata["true_source_distance_m"] = float(args.true_source_distance_m)
    if args.true_source_bearing_deg is not None:
        metadata["true_source_bearing_deg"] = float(args.true_source_bearing_deg)
    return metadata


def build_heartbeat(station: SimulatedStation, event: AcousticEvent, timestamp: float, sample_rate: int) -> StationHeartbeat:
    return StationHeartbeat(
        station_id=station.station_id,
        station_name=event.station_name,
        timestamp_unix=timestamp,
        status="online",
        station_location=event.station_location,
        audio_device="synthetic",
        sample_rate=sample_rate,
        channels=event.channel_count,
        calibrated=event.calibrated,
        detector_version=DETECTOR_VERSION,
        station_mode=event.station_mode,
        last_event_status=event.status.value if hasattr(event.status, "value") else str(event.status),
        last_harmonic_score=event.harmonic_score,
        last_hf_p_drone=event.hf_p_drone,
        metadata={"source": "simulate_station"},
    )


def make_simulated_stations(
    count: int,
    station_id: str,
    detector_config: StationDetectorStateConfig | None = None,
) -> list[SimulatedStation]:
    stations = []
    detector_config = detector_config or StationDetectorStateConfig()
    for idx in range(count):
        sim_id = station_id if count == 1 else f"sim_{idx + 1:03d}"
        stations.append(
            SimulatedStation(
                station_id=sim_id,
                detector_state=StationDetectorState(detector_config),
                station_index=idx,
                strength=max(0.45, 1.0 - idx * 0.18),
                delay_sec=idx * 2.0,
            )
        )
    return stations


def run_simulation(args: argparse.Namespace) -> None:
    stations = make_simulated_stations(args.num_stations, args.station_id)
    start = time.time()
    total_duration = 46.0 if args.scenario == "drone_pass" else 30.0
    heartbeat_url = heartbeat_url_from_events_url(args.server)
    last_heartbeat_by_station: dict[str, float] = {}

    while True:
        loop_start = time.time()
        elapsed = loop_start - start
        for station in stations:
            station_elapsed = max(0.0, elapsed - station.delay_sec)
            audio = generate_synthetic_audio(
                args.scenario,
                station_elapsed,
                sample_rate=args.sample_rate,
                window_sec=args.window_sec,
                channels=args.channels,
                station_index=station.station_index,
                strength=station.strength,
            )
            event = build_event(
                station,
                audio,
                args.sample_rate,
                loop_start,
                metadata_extra=_scenario_metadata(args),
            )
            requests.post(args.server, json=event.model_dump(mode="json"), timeout=2.0)
            if args.heartbeat and loop_start - last_heartbeat_by_station.get(station.station_id, 0.0) >= args.heartbeat_interval:
                heartbeat = build_heartbeat(station, event, loop_start, args.sample_rate)
                requests.post(heartbeat_url, json=heartbeat.model_dump(mode="json"), timeout=2.0)
                last_heartbeat_by_station[station.station_id] = loop_start
            print(
                f"{event.station_id} {event.status:11s} "
                f"conf={event.confidence:.2f} harm={event.harmonic_score:.1f} "
                f"f0={event.best_f0_hz} agree={event.channel_agreement_count}/{event.channel_count}"
            )

        if not args.realtime and elapsed >= total_duration:
            break
        if args.realtime:
            time.sleep(max(0.0, args.window_sec - (time.time() - loop_start)))
        elif elapsed < total_duration:
            start -= args.window_sec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080/events")
    parser.add_argument("--station-id", default="sim_001")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument(
        "--scenario",
        choices=[
            "background",
            "drone_hover",
            "drone_pass",
            "multi_rotor_jitter",
            "false_positive_fan",
            "motorcycle_like",
        ],
        default="drone_pass",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--num-stations", type=int, default=1)
    parser.add_argument("--heartbeat", action="store_true")
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--scenario-id")
    parser.add_argument("--simulated-source-id")
    parser.add_argument("--coverage-radius-m", type=float)
    parser.add_argument("--true-source-latitude", type=float)
    parser.add_argument("--true-source-longitude", type=float)
    parser.add_argument("--true-source-distance-m", type=float)
    parser.add_argument("--true-source-bearing-deg", type=float)
    return parser.parse_args()


def main() -> None:
    run_simulation(parse_args())


if __name__ == "__main__":
    main()
