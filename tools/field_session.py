from __future__ import annotations

import csv
import json
import math
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tools.eval_manifest_dataset import _is_alert, _is_candidate, _is_strong


FIELD_LABELS = {"drone", "background", "helicopter", "wind", "vehicle", "unknown"}
NOTES_FIELDNAMES = [
    "timestamp",
    "timestamp_unix",
    "iso_time",
    "label",
    "distance_m",
    "drone_model",
    "bearing_deg",
    "ground_truth_bearing_deg",
    "maneuver",
    "station_id",
    "note",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def session_id_from_time(now: datetime | None = None) -> str:
    now = utc_now() if now is None else now
    return now.strftime("%Y%m%d_%H%M%S")


def create_field_session(
    *,
    root: str | Path = "field_sessions",
    session_id: str | None = None,
    location: str = "",
    station_ids: list[str] | None = None,
    weather: str = "",
    wind_estimate: str = "",
    drone_model: str = "",
    operator_notes: str = "",
    now: datetime | None = None,
) -> Path:
    now = utc_now() if now is None else now
    session_id = session_id or session_id_from_time(now)
    session_dir = Path(root) / session_id
    for child in ("stations", "recordings", "reports"):
        (session_dir / child).mkdir(parents=True, exist_ok=True)

    payload = {
        "session_id": session_id,
        "created_unix": now.timestamp(),
        "date_time": now.isoformat(),
        "location": location,
        "station_ids": station_ids or [],
        "weather": weather,
        "wind_estimate": wind_estimate,
        "drone_model": drone_model,
        "operator_notes": operator_notes,
    }
    (session_dir / "session.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    ensure_notes_csv(session_dir / "notes.csv")
    return session_dir


def ensure_notes_csv(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NOTES_FIELDNAMES)
        writer.writeheader()


def append_field_note(
    *,
    session: str | Path,
    label: str,
    distance_m: float | None = None,
    drone_model: str = "",
    note: str = "",
    bearing_deg: float | None = None,
    ground_truth_bearing_deg: float | None = None,
    maneuver: str = "",
    station_id: str = "",
    timestamp_unix: float | None = None,
) -> dict[str, Any]:
    label = label.lower()
    if label not in FIELD_LABELS:
        raise ValueError(f"unsupported field label: {label}")
    timestamp_unix = time.time() if timestamp_unix is None else float(timestamp_unix)
    bearing_value = bearing_deg if bearing_deg is not None else ground_truth_bearing_deg
    row = {
        "timestamp": timestamp_unix,
        "timestamp_unix": timestamp_unix,
        "iso_time": datetime.fromtimestamp(timestamp_unix, timezone.utc).isoformat(),
        "label": label,
        "distance_m": distance_m,
        "drone_model": drone_model,
        "bearing_deg": bearing_value,
        "ground_truth_bearing_deg": bearing_value,
        "maneuver": maneuver,
        "station_id": station_id,
        "note": note,
    }
    notes_path = Path(session) / "notes.csv"
    ensure_notes_csv(notes_path)
    with notes_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NOTES_FIELDNAMES)
        writer.writerow(row)
    return row


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_prediction_rows(session: str | Path, extra_predictions: list[Path] | None = None) -> list[dict[str, Any]]:
    session = Path(session)
    paths = list(extra_predictions or [])
    paths.extend(sorted((session / "stations").glob("*.jsonl")))
    paths.extend(sorted((session / "reports").glob("*.jsonl")))
    paths.extend(sorted((session / "reports").glob("*.csv")))
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            rows.extend(read_jsonl_rows(path))
        elif path.suffix.lower() == ".csv":
            rows.extend(read_csv_rows(path))
    return rows


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _timestamp(row: dict[str, Any]) -> float | None:
    return _float_or_none(row.get("timestamp_unix") or row.get("timestamp") or row.get("created_unix"))


def _max_run(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _max_metric(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        for key in keys:
            value = _float_or_none(row.get(key))
            if value is not None:
                values.append(value)
                break
    return max(values) if values else None


def _angle_error_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def assign_rows_to_field_notes(
    prediction_rows: list[dict[str, Any]],
    notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    notes_with_ts = sorted((row for row in notes if _timestamp(row) is not None), key=lambda row: float(_timestamp(row)))
    if not notes_with_ts:
        return [dict(row) for row in prediction_rows]
    assigned = []
    for row in prediction_rows:
        ts = _timestamp(row)
        if ts is None:
            assigned.append(dict(row))
            continue
        matched = None
        for idx, note in enumerate(notes_with_ts):
            start = float(_timestamp(note))
            end = float(_timestamp(notes_with_ts[idx + 1])) if idx + 1 < len(notes_with_ts) else float("inf")
            if start <= ts < end:
                matched = note
                break
        merged = dict(row)
        if matched is not None:
            for key in ("label", "distance_m", "drone_model", "bearing_deg", "ground_truth_bearing_deg", "maneuver", "note"):
                if matched.get(key) not in (None, ""):
                    merged[key] = matched.get(key)
            merged["field_event_timestamp_unix"] = _timestamp(matched)
        assigned.append(merged)
    return assigned


def field_session_summary(
    *,
    notes: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = assign_rows_to_field_notes(prediction_rows, notes)
    by_distance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        distance = row.get("distance_m")
        key = str(distance if distance not in (None, "") else "unknown")
        by_distance[key].append(row)

    distance_summary: dict[str, dict[str, Any]] = {}
    for distance, items in sorted(by_distance.items()):
        ordered = sorted(items, key=lambda row: _timestamp(row) or 0.0)
        candidate_flags = [_is_candidate(row) for row in ordered]
        strong_flags = [_is_strong(row) for row in ordered]
        alert_flags = [_is_alert(row) for row in ordered]
        drone_rows = [row for row in ordered if str(row.get("label") or "").lower() == "drone"]
        first_candidate_time = next((_timestamp(row) for row in ordered if _is_candidate(row)), None)
        field_start = _timestamp(drone_rows[0]) if drone_rows else None
        bearing_errors = []
        for row in ordered:
            truth = _float_or_none(row.get("ground_truth_bearing_deg") or row.get("bearing_deg"))
            estimated = _float_or_none(row.get("estimated_azimuth_deg") or row.get("estimated_bearing_deg"))
            if truth is not None and estimated is not None:
                bearing_errors.append(_angle_error_deg(estimated, truth))
        distance_summary[distance] = {
            "windows": len(ordered),
            "candidate_any": any(candidate_flags),
            "candidate_run2": _max_run(candidate_flags) >= 2,
            "candidate_run3": _max_run(candidate_flags) >= 3,
            "strong_any": any(strong_flags),
            "strong_run2": _max_run(strong_flags) >= 2,
            "strong_run3": _max_run(strong_flags) >= 3,
            "alert_any": any(alert_flags),
            "detection_delay_sec": (
                max(0.0, first_candidate_time - field_start)
                if first_candidate_time is not None and field_start is not None
                else None
            ),
            "max_ml": _max_metric(ordered, "ml_drone_pct", "hf_p_drone"),
            "max_harmonic": _max_metric(ordered, "harmonic_evidence_pct_smoothed", "harmonic_evidence_pct", "harmonic_score"),
            "max_combined": _max_metric(ordered, "combined_drone_evidence_pct"),
            "bearing_error_deg_median": _median(bearing_errors),
        }

    non_drone_rows = [row for row in rows if str(row.get("label") or "").lower() not in {"drone", "unknown"}]
    non_drone_candidate_rows = [row for row in non_drone_rows if _is_candidate(row)]
    non_drone_alert_rows = [row for row in non_drone_rows if _is_alert(row)]
    non_drone_hours = _duration_hours(non_drone_rows)
    recommendations = []
    if non_drone_hours and len(non_drone_candidate_rows) / non_drone_hours > 2.0:
        recommendations.append("False candidate rate is high; consider increasing ML persistence or combined evidence thresholds.")
    if non_drone_alert_rows:
        recommendations.append("False alerts occurred; inspect harmonic-only gating, clipping, and non-drone tonal sources.")
    if not rows:
        recommendations.append("No prediction rows found; verify station history/report files were copied into the session.")

    return {
        "notes": len(notes),
        "windows": len(rows),
        "distance_summary": distance_summary,
        "false_positives_per_hour": len(non_drone_candidate_rows) / non_drone_hours if non_drone_hours else 0.0,
        "false_alerts_per_hour": len(non_drone_alert_rows) / non_drone_hours if non_drone_hours else 0.0,
        "recommended_threshold_adjustments": recommendations,
    }


def _duration_hours(rows: list[dict[str, Any]]) -> float:
    timestamps = [_timestamp(row) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    if len(timestamps) >= 2:
        return max(0.0, max(timestamps) - min(timestamps)) / 3600.0
    window_secs = [_float_or_none(row.get("window_sec") or row.get("duration_sec")) for row in rows]
    window_secs = [value for value in window_secs if value is not None and value > 0]
    return sum(window_secs) / 3600.0


def _median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def save_debug_capture(
    *,
    state_path: str | Path,
    history_path: str | Path | None = None,
    output_root: str | Path = "field_debug_captures",
    seconds: float = 30.0,
    label: str = "unknown",
    note: str = "",
    now: float | None = None,
) -> Path:
    state_path = Path(state_path)
    if not state_path.exists():
        raise FileNotFoundError(f"local monitor state not found: {state_path}")
    now = time.time() if now is None else float(now)
    label = label.lower()
    capture_id = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%d_%H%M%S")
    capture_dir = Path(output_root) / f"{capture_id}_{label}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state_path, capture_dir / "latest.json")

    if history_path is not None and Path(history_path).exists():
        rows = read_jsonl_rows(history_path)
        recent = [row for row in rows if (_timestamp(row) is None or now - float(_timestamp(row)) <= float(seconds))]
        with (capture_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
            for row in recent:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    metadata = {
        "created_unix": now,
        "label": label,
        "note": note,
        "seconds": float(seconds),
        "state_path": str(state_path),
        "history_path": str(history_path) if history_path is not None else None,
    }
    (capture_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return capture_dir
