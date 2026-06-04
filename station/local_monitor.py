from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from shared.event_schema import AcousticEvent


def decimated_waveform(mono: np.ndarray, max_points: int = 1200) -> list[float]:
    audio = np.asarray(mono, dtype=np.float32).reshape(-1)
    max_points = max(1, int(max_points))
    if audio.size <= max_points:
        return [float(x) for x in audio]
    indices = np.linspace(0, audio.size - 1, max_points).astype(int)
    return [float(x) for x in audio[indices]]


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def append_history_jsonl(path: str | Path, row: Mapping[str, Any], max_rows: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if max_rows is not None and max_rows > 0 and path.exists():
        rows = path.read_text(encoding="utf-8").splitlines()[-(int(max_rows) - 1) :]
        rows.append(json.dumps(dict(row), separators=(",", ":"), sort_keys=True))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), separators=(",", ":"), sort_keys=True) + "\n")


def local_monitor_paths(
    cfg: Mapping[str, Any],
    station_id: str,
) -> tuple[Path, Path]:
    local_cfg = cfg.get("local_monitor", {}) if isinstance(cfg, Mapping) else {}
    directory = Path(str(local_cfg.get("directory", "runtime/stations")))
    state_path = Path(str(local_cfg.get("state_path") or directory / f"{station_id}_latest.json"))
    history_path = Path(str(local_cfg.get("history_path") or directory / f"{station_id}_history.jsonl"))
    return state_path, history_path


def history_row_from_event(event: AcousticEvent) -> dict[str, Any]:
    metadata = event.metadata or {}
    status = event.status.value if hasattr(event.status, "value") else str(event.status)
    return {
        "timestamp": event.timestamp_unix,
        "status": status,
        "operator_label": event.operator_label or metadata.get("operator_label"),
        "confidence": event.confidence,
        "harmonic_score": event.harmonic_score,
        "harmonic_evidence_pct_smoothed": event.harmonic_evidence_pct_smoothed
        if event.harmonic_evidence_pct_smoothed is not None
        else metadata.get("harmonic_evidence_pct_smoothed"),
        "ml_drone_pct": event.ml_drone_pct if event.ml_drone_pct is not None else metadata.get("ml_drone_pct"),
        "combined_drone_evidence_pct": event.combined_drone_evidence_pct
        if event.combined_drone_evidence_pct is not None
        else metadata.get("combined_drone_evidence_pct"),
        "best_f0_hz": event.best_f0_hz,
        "rms": event.rms,
        "candidate_run": event.candidate_run if event.candidate_run is not None else metadata.get("candidate_run"),
        "ml_positive_run": event.ml_positive_run if event.ml_positive_run is not None else metadata.get("ml_positive_run"),
        "strong_run": event.strong_run if event.strong_run is not None else metadata.get("strong_run"),
        "decision_stage": event.decision_stage if event.decision_stage is not None else metadata.get("decision_stage"),
        "blocked_by": event.blocked_by if event.blocked_by is not None else metadata.get("blocked_by"),
        "candidate_block_reason": event.candidate_block_reason
        if event.candidate_block_reason is not None
        else metadata.get("candidate_block_reason"),
        "alert_block_reason": event.alert_block_reason
        if event.alert_block_reason is not None
        else metadata.get("alert_block_reason"),
        "alert_blocked_reason": event.alert_blocked_reason
        if event.alert_blocked_reason is not None
        else metadata.get("alert_blocked_reason"),
        "why_candidate_run_reset": event.why_candidate_run_reset
        if event.why_candidate_run_reset is not None
        else metadata.get("why_candidate_run_reset"),
    }


def build_local_monitor_snapshot(
    *,
    event: AcousticEvent,
    mono: np.ndarray,
    waveform_points: int,
    spectrum: Mapping[str, Any],
    spectrogram: Mapping[str, Any],
    harmonic_lines: list[dict[str, Any]],
    hf_result: Any = None,
    beam_result: Any = None,
    server_state: Mapping[str, Any] | None = None,
    updated_unix: float | None = None,
) -> dict[str, Any]:
    metadata = event.metadata or {}
    updated_unix = time.time() if updated_unix is None else float(updated_unix)
    beam = {
        "estimated_azimuth_deg": event.estimated_azimuth_deg,
        "direction_confidence": event.direction_confidence,
        "beamforming_method": event.beamforming_method or metadata.get("beamforming_method"),
        "beam_score": event.beam_score if event.beam_score is not None else metadata.get("beam_score"),
        "beam_snr_gain_db": event.beam_snr_gain_db if event.beam_snr_gain_db is not None else metadata.get("beam_snr_gain_db"),
        "beam_confidence_pct": event.beam_confidence_pct
        if getattr(event, "beam_confidence_pct", None) is not None
        else metadata.get("beam_confidence_pct"),
        "beam_peak_to_median": event.beam_peak_to_median
        if getattr(event, "beam_peak_to_median", None) is not None
        else metadata.get("beam_peak_to_median"),
        "beam_peak_to_second_peak": event.beam_peak_to_second_peak
        if getattr(event, "beam_peak_to_second_peak", None) is not None
        else metadata.get("beam_peak_to_second_peak"),
        "raw_estimated_azimuth_deg": metadata.get("raw_estimated_azimuth_deg"),
        "second_peak_bearing_deg": event.second_peak_bearing_deg
        if getattr(event, "second_peak_bearing_deg", None) is not None
        else metadata.get("second_peak_bearing_deg"),
        "second_peak_ratio": event.second_peak_ratio
        if getattr(event, "second_peak_ratio", None) is not None
        else metadata.get("second_peak_ratio"),
        "peak_ratio": event.peak_ratio if getattr(event, "peak_ratio", None) is not None else metadata.get("peak_ratio"),
        "bearing_ambiguity_deg": event.bearing_ambiguity_deg
        if getattr(event, "bearing_ambiguity_deg", None) is not None
        else metadata.get("bearing_ambiguity_deg"),
        "bearing_reliable": event.bearing_reliable
        if getattr(event, "bearing_reliable", None) is not None
        else metadata.get("bearing_reliable"),
        "bearing_reject_reason": event.bearing_reject_reason
        if getattr(event, "bearing_reject_reason", None) is not None
        else metadata.get("bearing_reject_reason"),
        "bearing_quality": event.bearing_quality
        if getattr(event, "bearing_quality", None) is not None
        else metadata.get("bearing_quality"),
        "bearing_stable": event.bearing_stable if event.bearing_stable is not None else metadata.get("bearing_stable"),
        "bearing_uncertainty_deg": event.bearing_uncertainty_deg
        if event.bearing_uncertainty_deg is not None
        else metadata.get("bearing_uncertainty_deg"),
    }
    if beam_result is not None:
        beam["beam_scan_deg"] = getattr(beam_result, "beam_scan_deg", None)
        beam["beam_scan_score"] = getattr(beam_result, "beam_scan_score", None)

    return {
        "updated_unix": updated_unix,
        "event": event.model_dump(mode="json"),
        "audio": {
            "waveform": decimated_waveform(mono, waveform_points),
            "waveform_points": int(waveform_points),
            "source_sample_count": int(np.asarray(mono).size),
            "sample_rate": metadata.get("sample_rate"),
            "channel_rms": metadata.get("channel_rms"),
            "channel_health": metadata.get("channel_health"),
            "bad_channels": metadata.get("bad_channels"),
            "kept_channels": metadata.get("kept_channels"),
            "calibration_loaded": metadata.get("calibration_loaded"),
            "calibration_file": metadata.get("calibration_file"),
        },
        "spectrum": dict(spectrum),
        "spectrogram": dict(spectrogram),
        "harmonic_lines": harmonic_lines,
        "hf": {
            "p_drone": event.hf_p_drone,
            "label": getattr(hf_result, "label", None) if hf_result is not None else metadata.get("hf_label"),
            "class_probs": getattr(hf_result, "class_probs", {}) if hf_result is not None else metadata.get("hf_class_probs", {}),
            "error": getattr(hf_result, "error", None) if hf_result is not None else metadata.get("hf_error_message"),
            "available": not bool(event.hf_error),
        },
        "beam": beam,
        "server": dict(server_state or {}),
    }


def write_local_monitor_snapshot(
    *,
    state_path: str | Path,
    history_path: str | Path,
    snapshot: Mapping[str, Any],
    history_row: Mapping[str, Any],
    history_max_rows: int | None = None,
) -> None:
    atomic_write_json(state_path, snapshot)
    append_history_jsonl(history_path, history_row, max_rows=history_max_rows)


def is_stale_state(snapshot: Mapping[str, Any], stale_after_sec: float = 3.0, now: float | None = None) -> bool:
    updated = snapshot.get("updated_unix")
    if updated is None:
        return True
    now = time.time() if now is None else float(now)
    return now - float(updated) > float(stale_after_sec)
