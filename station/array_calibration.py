from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from station.direction import fractional_delay


@dataclass
class ArrayCalibration:
    channel_count: int
    input_channel_order: list[int]
    gain_correction: list[float]
    delay_correction_samples: list[float]
    bad_channels: list[int]
    channel_rms: list[float]
    channel_health: list[str]
    calibration_valid: bool = True
    calibration_type: str = "measured"
    source_path: str | None = None


def channel_rms(audio: np.ndarray) -> list[float]:
    array = _as_2d(audio)
    return [float(np.sqrt(np.mean(array[:, idx].astype(np.float64) ** 2))) for idx in range(array.shape[1])]


def detect_strongest_channel(audio: np.ndarray) -> int:
    array = _as_2d(audio)
    peaks = np.max(np.abs(array), axis=0)
    return int(np.argmax(peaks))


def verify_channel_order(detected_channels: list[int], channel_count: int) -> dict[str, Any]:
    expected = list(range(int(channel_count)))
    detected = [int(item) for item in detected_channels]
    missing = [idx for idx in expected if idx not in detected]
    duplicates = sorted({idx for idx in detected if detected.count(idx) > 1})
    correct = detected == expected and not missing and not duplicates
    return {
        "channel_count": int(channel_count),
        "expected_input_channel_order": expected,
        "detected_input_channel_order": detected,
        "correct": bool(correct),
        "missing_channels": missing,
        "duplicate_channels": duplicates,
        "suggested_input_channel_order": detected if len(detected) == int(channel_count) else expected,
    }


def estimate_gain_correction(audio: np.ndarray, *, min_rms: float = 1e-8) -> tuple[list[float], list[float], list[str]]:
    rms = channel_rms(audio)
    good = [value for value in rms if value >= float(min_rms)]
    target = float(np.median(good)) if good else 1.0
    gains = [float(target / max(value, float(min_rms))) for value in rms]
    health = ["dead" if value < float(min_rms) else "ok" for value in rms]
    return gains, rms, health


def estimate_delay_correction_samples(
    audio: np.ndarray,
    *,
    reference_channel: int = 0,
    max_lag_samples: int = 256,
) -> list[float]:
    array = _as_2d(audio).astype(np.float64)
    reference_channel = int(reference_channel)
    reference = _center(array[:, reference_channel])
    corrections: list[float] = []
    for idx in range(array.shape[1]):
        if idx == reference_channel:
            corrections.append(0.0)
            continue
        current = _center(array[:, idx])
        corr = np.correlate(current, reference, mode="full")
        center = len(reference) - 1
        lo = max(0, center - int(max_lag_samples))
        hi = min(corr.size, center + int(max_lag_samples) + 1)
        lag = int(np.argmax(corr[lo:hi]) + lo - center)
        corrections.append(float(-lag))
    return corrections


def build_calibration(
    audio: np.ndarray,
    *,
    sample_rate: int,
    input_channel_order: list[int] | None = None,
    bad_channels: list[int] | None = None,
    min_rms: float = 1e-6,
    dropout_ratio: float = 0.15,
    estimate_delay: bool = True,
    max_lag_samples: int = 256,
) -> dict[str, Any]:
    array = _as_2d(audio)
    channels = int(array.shape[1])
    order = [int(idx) for idx in (input_channel_order or list(range(channels)))]
    ordered = array[:, order]
    gains, rms, health = estimate_gain_correction(ordered, min_rms=min_rms)
    median_rms = float(np.median([value for value in rms if value >= min_rms] or [0.0]))
    detected_bad = set(int(idx) for idx in (bad_channels or []))
    for idx, value in enumerate(rms):
        if value < min_rms or (median_rms > 0 and value < median_rms * float(dropout_ratio)):
            detected_bad.add(idx)
            health[idx] = "dropout" if value >= min_rms else "silent"
    calibration_valid = bool(channels > 0 and any(value > float(min_rms) for value in rms) and not detected_bad)
    delays = (
        estimate_delay_correction_samples(ordered, max_lag_samples=max_lag_samples)
        if estimate_delay and channels >= 2
        else [0.0] * channels
    )
    return {
        "version": 1,
        "created_unix": time.time(),
        "sample_rate": int(sample_rate),
        "channel_count": channels,
        "input_channel_order": order,
        "channel_permutation": order,
        "gain_correction": gains,
        "delay_correction_samples": delays,
        "bad_channels": sorted(detected_bad),
        "channel_rms": rms,
        "channel_health": health,
        "calibration_valid": calibration_valid,
        "calibration_type": "measured",
    }


def load_calibration(path: str | Path | None) -> ArrayCalibration | None:
    if not path:
        return None
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    channel_count = int(payload.get("channel_count") or len(payload.get("gain_correction") or []))
    order = payload.get("input_channel_order") or payload.get("channel_permutation") or list(range(channel_count))
    gains = payload.get("gain_correction") or [1.0] * len(order)
    delays = payload.get("delay_correction_samples") or [0.0] * len(order)
    bad = payload.get("bad_channels") or payload.get("dead_channels") or []
    rms = [float(value) for value in (payload.get("channel_rms") or [0.0] * len(order))]
    bad = [int(idx) for idx in bad]
    health = _normalized_channel_health(payload.get("channel_health") or ["ok"] * len(order), rms, bad)
    explicit_valid = payload.get("calibration_valid")
    calibration_valid = validate_calibration_payload({**payload, "channel_rms": rms, "channel_health": health, "bad_channels": bad})[
        "calibration_valid"
    ]
    if explicit_valid is not None:
        calibration_valid = bool(explicit_valid) and calibration_valid
    return ArrayCalibration(
        channel_count=channel_count,
        input_channel_order=[int(idx) for idx in order],
        gain_correction=[float(value) for value in gains],
        delay_correction_samples=[float(value) for value in delays],
        bad_channels=bad,
        channel_rms=rms,
        channel_health=health,
        calibration_valid=calibration_valid,
        calibration_type=str(payload.get("calibration_type") or ("measured" if calibration_valid else "placeholder")),
        source_path=str(path),
    )


def apply_array_calibration(
    audio: np.ndarray,
    mic_positions_m: np.ndarray | None = None,
    calibration: ArrayCalibration | dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    array = _as_2d(audio)
    if calibration is None:
        return array.astype(np.float32), mic_positions_m, _calibration_metadata(None, channel_rms(array))
    if isinstance(calibration, dict):
        calibration = _calibration_from_dict(calibration)
    if not calibration.calibration_valid:
        return array.astype(np.float32), mic_positions_m, _calibration_metadata(calibration, channel_rms(array))
    if max(calibration.input_channel_order, default=-1) >= array.shape[1]:
        raise ValueError("calibration input_channel_order references a missing audio channel")

    corrected = array[:, calibration.input_channel_order].astype(np.float32, copy=True)
    gains = _pad(calibration.gain_correction, corrected.shape[1], 1.0)
    delays = _pad(calibration.delay_correction_samples, corrected.shape[1], 0.0)
    for idx in range(corrected.shape[1]):
        corrected[:, idx] *= float(gains[idx])
        if abs(float(delays[idx])) > 1e-9:
            corrected[:, idx] = fractional_delay(corrected[:, idx], float(delays[idx])).astype(np.float32)

    keep = [idx for idx in range(corrected.shape[1]) if idx not in set(calibration.bad_channels)]
    positions = None if mic_positions_m is None else np.asarray(mic_positions_m, dtype=np.float64)
    if positions is not None:
        positions = positions[keep, :]
    corrected = corrected[:, keep]
    meta = _calibration_metadata(calibration, channel_rms(corrected), kept_channels=keep)
    return corrected.astype(np.float32), positions, meta


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_2d(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"audio must be mono or multi-channel 2D, got shape {array.shape}")
    return array


def _center(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - float(np.mean(values))
    std = float(np.std(centered))
    return centered / std if std > 1e-12 else centered


def _pad(values: list[float], size: int, fill: float) -> list[float]:
    padded = list(values[:size])
    padded.extend([float(fill)] * (int(size) - len(padded)))
    return padded


def _calibration_from_dict(payload: dict[str, Any]) -> ArrayCalibration:
    path = payload.get("source_path")
    channel_count = int(payload.get("channel_count") or len(payload.get("input_channel_order") or []))
    order = payload.get("input_channel_order") or payload.get("channel_permutation") or list(range(channel_count))
    rms = [float(value) for value in payload.get("channel_rms", [0.0] * len(order))]
    bad = [int(idx) for idx in payload.get("bad_channels", [])]
    health = _normalized_channel_health(payload.get("channel_health", ["ok"] * len(order)), rms, bad)
    valid = validate_calibration_payload({**payload, "channel_rms": rms, "channel_health": health, "bad_channels": bad})[
        "calibration_valid"
    ]
    if payload.get("calibration_valid") is not None:
        valid = bool(payload.get("calibration_valid")) and valid
    return ArrayCalibration(
        channel_count=channel_count,
        input_channel_order=[int(idx) for idx in order],
        gain_correction=[float(value) for value in payload.get("gain_correction", [1.0] * len(order))],
        delay_correction_samples=[
            float(value) for value in payload.get("delay_correction_samples", [0.0] * len(order))
        ],
        bad_channels=bad,
        channel_rms=rms,
        channel_health=health,
        calibration_valid=valid,
        calibration_type=str(payload.get("calibration_type") or ("measured" if valid else "placeholder")),
        source_path=None if path is None else str(path),
    )


def _calibration_metadata(
    calibration: ArrayCalibration | None,
    rms: list[float],
    *,
    kept_channels: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "calibration_loaded": calibration is not None,
        "calibration_file": None if calibration is None else calibration.source_path,
        "calibration_valid": False if calibration is None else calibration.calibration_valid,
        "calibration_type": None if calibration is None else calibration.calibration_type,
        "channel_rms": rms,
        "channel_health": [] if calibration is None else calibration.channel_health,
        "bad_channels": [] if calibration is None else calibration.bad_channels,
        "kept_channels": list(range(len(rms))) if kept_channels is None else kept_channels,
        "input_channel_order": None if calibration is None else calibration.input_channel_order,
    }


def validate_calibration_payload(payload: dict[str, Any], *, min_rms: float = 1e-8) -> dict[str, Any]:
    channel_count = int(payload.get("channel_count") or len(payload.get("channel_rms") or []))
    rms = [float(value) for value in (payload.get("channel_rms") or [0.0] * channel_count)]
    bad = [int(idx) for idx in payload.get("bad_channels", [])]
    health = _normalized_channel_health(payload.get("channel_health") or ["ok"] * len(rms), rms, bad, min_rms=min_rms)
    has_signal = any(value > float(min_rms) for value in rms)
    silent = [idx for idx, value in enumerate(rms) if value <= float(min_rms)]
    valid = bool(channel_count > 0 and has_signal and not silent)
    return {
        "calibration_valid": valid,
        "channel_health": health,
        "silent_channels": silent,
    }


def _normalized_channel_health(
    health: list[str],
    rms: list[float],
    bad_channels: list[int],
    *,
    min_rms: float = 1e-8,
) -> list[str]:
    out = [str(value) for value in list(health)]
    if len(out) < len(rms):
        out.extend(["unknown"] * (len(rms) - len(out)))
    out = out[: len(rms)]
    bad = set(int(idx) for idx in bad_channels)
    for idx, value in enumerate(rms):
        if idx in bad:
            out[idx] = "bad" if out[idx] == "ok" else out[idx]
        if float(value) <= float(min_rms):
            out[idx] = "silent"
    return out
