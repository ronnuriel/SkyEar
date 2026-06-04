from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from station.array_profiles import array_profile, circular_mic_positions
from station.beamforming import bearing_quality_from_result, estimate_bearing
from station.station_agent import apply_mic_array_profile_defaults


def angle_error_deg(predicted: float | None, truth: float | None) -> float | None:
    if predicted is None or truth is None:
        return None
    return abs((float(predicted) - float(truth) + 180.0) % 360.0 - 180.0)


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
        return int(sample_rate), np.asarray(audio, dtype=np.float32)
    except Exception:
        from scipy.io import wavfile

        sample_rate, audio = wavfile.read(str(path))
        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        if np.issubdtype(audio.dtype, np.integer):
            max_value = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / max(max_value, 1.0)
        return int(sample_rate), audio.astype(np.float32)


def load_truth(path: str | Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "bearing_deg": float(row["bearing_deg"]),
                }
            )
    return rows


def load_truth_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path).with_suffix(".metadata.json")
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def geometry_warning_from_metadata(metadata: dict[str, Any]) -> str:
    actual = metadata.get("actual_mic_positions_m")
    nominal = metadata.get("nominal_mic_positions_m")
    if not actual or not nominal:
        return ""
    actual_array = np.asarray(actual, dtype=np.float64)
    nominal_array = np.asarray(nominal, dtype=np.float64)
    if actual_array.shape != nominal_array.shape:
        return "simulated_geometry_metadata_mismatch"
    max_error_m = float(np.max(np.linalg.norm(actual_array[:, :2] - nominal_array[:, :2], axis=1)))
    if max_error_m >= 0.03:
        return f"simulated_mic_position_mismatch_max_cm={max_error_m * 100.0:.1f}"
    return ""


def truth_bearing_at(rows: list[dict[str, float]], timestamp_sec: float) -> float | None:
    if not rows:
        return None
    times = np.asarray([row["timestamp_sec"] for row in rows], dtype=np.float64)
    bearings = np.asarray([row["bearing_deg"] for row in rows], dtype=np.float64)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(bearings)))
    truth = float(np.interp(float(timestamp_sec), times, unwrapped))
    return truth % 360.0


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return apply_mic_array_profile_defaults(cfg)


def _mic_positions_from_config(cfg: dict[str, Any]) -> np.ndarray | None:
    positions = (cfg.get("mic_array") or {}).get("mic_positions_m")
    if not positions:
        return None
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    return array[:, :3] if array.shape[1] >= 3 else np.pad(array, ((0, 0), (0, 1)))


def _override_positions(
    positions: np.ndarray | None,
    *,
    channel_count: int,
    profile_override: str | None = None,
    array_radius_m: float | None = None,
) -> tuple[np.ndarray | None, str]:
    source = "config"
    if profile_override:
        profile = array_profile(str(profile_override))
        if profile is None or not profile.get("mic_positions_m"):
            raise ValueError(f"Unknown or non-array eval profile: {profile_override}")
        positions = np.asarray(profile["mic_positions_m"], dtype=np.float64)
        source = f"profile:{profile_override}"
    if array_radius_m is not None:
        positions = np.asarray(circular_mic_positions(int(channel_count), float(array_radius_m)), dtype=np.float64)
        source = f"radius:{float(array_radius_m):.3f}m"
    if positions is None:
        return None, source
    if positions.ndim != 2 or positions.shape[1] < 2:
        return None, source
    positions = positions[:, :3] if positions.shape[1] >= 3 else np.pad(positions, ((0, 0), (0, 1)))
    return positions, source


def evaluate_array_audio(
    *,
    wav_path: str | Path,
    truth_path: str | Path,
    config_path: str | Path,
    window_sec: float,
    profile_override: str | None = None,
    array_radius_m: float | None = None,
) -> list[dict[str, Any]]:
    sample_rate, audio = read_wav(wav_path)
    truth = load_truth(truth_path)
    truth_metadata = load_truth_metadata(truth_path)
    geometry_warning = geometry_warning_from_metadata(truth_metadata)
    cfg = _load_config(config_path)
    positions, position_source = _override_positions(
        _mic_positions_from_config(cfg),
        channel_count=audio.shape[1],
        profile_override=profile_override,
        array_radius_m=array_radius_m,
    )
    beam_cfg = cfg.get("beamforming", {})
    dir_cfg = cfg.get("direction", {})
    det_cfg = cfg.get("detector", {})
    expected_channels = int((cfg.get("audio") or {}).get("channels", audio.shape[1]))
    beamforming_enabled = bool(beam_cfg.get("enabled", True))
    can_beamform = (
        beamforming_enabled
        and audio.ndim == 2
        and audio.shape[1] >= 2
        and positions is not None
        and positions.shape[0] == audio.shape[1]
        and expected_channels == audio.shape[1]
    )
    window_samples = max(1, int(round(float(window_sec) * int(sample_rate))))
    rows: list[dict[str, Any]] = []
    for window_idx, start in enumerate(range(0, audio.shape[0] - window_samples + 1, window_samples)):
        end = start + window_samples
        timestamp_sec = start / float(sample_rate)
        truth_bearing = truth_bearing_at(truth, timestamp_sec + float(window_sec) / 2.0)
        result = None
        if can_beamform:
            result = estimate_bearing(
                audio[start:end, :],
                int(sample_rate),
                positions,
                method=str(beam_cfg.get("method", "delay_and_sum")),
                scan_step_deg=int(beam_cfg.get("scan_step_deg", dir_cfg.get("scan_step_deg", 5))),
                low_hz=int(beam_cfg.get("low_hz", det_cfg.get("f0_min", 500))),
                high_hz=int(beam_cfg.get("high_hz", det_cfg.get("max_freq", 3000))),
                bearing_stability_deg=float(beam_cfg.get("bearing_stability_deg", 15.0)),
                min_beam_confidence_pct=float(
                    dir_cfg.get(
                        "min_beam_confidence_pct",
                        beam_cfg.get("min_beam_confidence_pct", 0.55),
                    )
                ),
                min_peak_ratio=float(dir_cfg.get("min_peak_ratio", beam_cfg.get("min_peak_ratio", 1.3))),
                max_second_peak_ratio=float(
                    dir_cfg.get(
                        "max_second_peak_ratio",
                        beam_cfg.get("max_second_peak_ratio", 0.85),
                    )
                ),
                reject_ambiguous_bearing=bool(
                    dir_cfg.get(
                        "reject_ambiguous_bearing",
                        beam_cfg.get("reject_ambiguous_bearing", True),
                    )
                ),
            )
        predicted = None if result is None else result.bearing_deg
        error = angle_error_deg(predicted, truth_bearing)
        rows.append(
            {
                "window_idx": window_idx,
                "timestamp_sec": timestamp_sec,
                "truth_bearing_deg": truth_bearing,
                "predicted_bearing_deg": predicted,
                "bearing_error_deg": error,
                "beam_confidence": None if result is None else result.beam_confidence_pct,
                "beam_score": None if result is None else result.beam_score,
                "beam_peak_to_median": None if result is None else result.beam_peak_to_median,
                "beam_peak_to_second_peak": None if result is None else result.beam_peak_to_second_peak,
                "second_peak_bearing_deg": None if result is None else result.second_peak_bearing_deg,
                "second_peak_ratio": None if result is None else result.second_peak_ratio,
                "peak_ratio": None if result is None else result.peak_ratio,
                "bearing_ambiguity_deg": None if result is None else result.bearing_ambiguity_deg,
                "bearing_reliable": False if result is None else result.bearing_reliable,
                "bearing_reject_reason": None if result is None else result.bearing_reject_reason,
                "bearing_quality": bearing_quality_from_result(result),
                "geometry_warning": geometry_warning,
                "bearing_stable": False if result is None else result.bearing_stable,
                "beamforming_attempted": bool(can_beamform),
                "eval_position_source": position_source,
            }
        )
    return rows


def write_eval_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_idx",
        "timestamp_sec",
        "truth_bearing_deg",
        "predicted_bearing_deg",
        "bearing_error_deg",
        "beam_confidence",
        "beam_score",
        "beam_peak_to_median",
        "beam_peak_to_second_peak",
        "second_peak_bearing_deg",
        "second_peak_ratio",
        "peak_ratio",
        "bearing_ambiguity_deg",
        "bearing_reliable",
        "bearing_reject_reason",
        "bearing_quality",
        "geometry_warning",
        "bearing_stable",
        "beamforming_attempted",
        "eval_position_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    errors = [float(row["bearing_error_deg"]) for row in rows if row.get("bearing_error_deg") is not None]
    detections = [row for row in rows if row.get("predicted_bearing_deg") is not None]
    reliable = [row for row in rows if row.get("bearing_reliable") is True or row.get("bearing_reliable") == "True"]
    confidences = [
        float(row["beam_confidence"])
        for row in rows
        if row.get("beam_confidence") is not None and row.get("beam_confidence") != ""
    ]
    if not errors:
        return {
            "median_error_deg": float("nan"),
            "p90_error_deg": float("nan"),
            "detection_rate": 0.0,
            "reliable_rate": 0.0,
            "median_confidence": float("nan"),
        }
    return {
        "median_error_deg": float(np.median(np.asarray(errors, dtype=np.float64))),
        "p90_error_deg": float(np.percentile(np.asarray(errors, dtype=np.float64), 90)),
        "detection_rate": float(len(detections) / max(1, len(rows))),
        "reliable_rate": float(len(reliable) / max(1, len(rows))),
        "median_confidence": float(np.median(np.asarray(confidences, dtype=np.float64))) if confidences else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate beamforming on synthetic or recorded array audio.")
    parser.add_argument("--wav", default="reports/sim/moving_source_8ch.wav")
    parser.add_argument("--truth", default="reports/sim/moving_source_truth.csv")
    parser.add_argument("--config", default="configs/config_station_array_8ch.yaml")
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--profile", help="Override eval mic array profile to intentionally test config mismatch.")
    parser.add_argument("--array-radius-m", type=float, help="Override eval array radius to intentionally test config mismatch.")
    parser.add_argument("--output", default="reports/sim/beam_eval.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_array_audio(
        wav_path=args.wav,
        truth_path=args.truth,
        config_path=args.config,
        window_sec=float(args.window_sec),
        profile_override=args.profile,
        array_radius_m=args.array_radius_m,
    )
    write_eval_csv(args.output, rows)
    metrics = summarize(rows)
    print(
        f"wrote {args.output} windows={len(rows)} "
        f"median_error_deg={metrics['median_error_deg']:.2f} "
        f"p90_error_deg={metrics['p90_error_deg']:.2f} "
        f"detection_rate={metrics['detection_rate']:.2f} "
        f"reliable_rate={metrics['reliable_rate']:.2f} "
        f"median_confidence={metrics['median_confidence']:.2f}"
    )


if __name__ == "__main__":
    main()
