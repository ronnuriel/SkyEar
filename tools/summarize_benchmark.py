from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.eval_manifest_dataset import _is_alert, _is_candidate, _is_strong


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_report(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _max_run(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _run_count(flags: list[bool]) -> int:
    count = 0
    previous = False
    for flag in flags:
        if flag and not previous:
            count += 1
        previous = flag
    return count


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _group_file(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("audio_path") or row.get("file_path") or row.get("file_name") or "unknown")].append(row)
    return grouped


def file_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    files = []
    for file_path, items in _group_file(rows).items():
        ordered = sorted(items, key=lambda row: int(float(row.get("window_idx") or 0)))
        candidate = [_is_candidate(row) for row in ordered]
        strong = [_is_strong(row) for row in ordered]
        alert = [_is_alert(row) for row in ordered]
        window_secs = [_float(row.get("window_sec")) or 1.0 for row in ordered]
        duration_sec = sum(window_secs)
        first_candidate = next(
            ((_float(row.get("window_idx")) or 0.0) * (_float(row.get("window_sec")) or 1.0) for row in ordered if _is_candidate(row)),
            None,
        )
        first_strong = next(
            ((_float(row.get("window_idx")) or 0.0) * (_float(row.get("window_sec")) or 1.0) for row in ordered if _is_strong(row)),
            None,
        )
        max_candidate_run = _max_run(candidate)
        max_strong_run = _max_run(strong)
        files.append(
            {
                "file_path": file_path,
                "dataset_id": ordered[0].get("dataset_id") or "unknown",
                "source_dataset": ordered[0].get("source_dataset") or ordered[0].get("dataset_id") or "unknown",
                "label": ordered[0].get("label") or "unknown",
                "distance_category": ordered[0].get("distance_category") or "unknown",
                "distance_m": ordered[0].get("distance_m") or "unknown",
                "environment": ordered[0].get("environment") or "unknown",
                "candidate_any": any(candidate),
                "candidate_run2": max_candidate_run >= 2,
                "candidate_run3": max_candidate_run >= 3,
                "strong_any": any(strong),
                "strong_run2": max_strong_run >= 2,
                "strong_run3": max_strong_run >= 3,
                "alert_any": any(alert),
                "time_to_first_candidate_sec": first_candidate,
                "time_to_first_strong_sec": first_strong,
                "max_ml": max((_float(row.get("ml_drone_pct") or row.get("hf_p_drone")) or 0.0 for row in ordered), default=0.0),
                "max_harmonic": max(
                    (_float(row.get("harmonic_evidence_pct_smoothed") or row.get("harmonic_evidence_pct")) or 0.0 for row in ordered),
                    default=0.0,
                ),
                "max_combined": max((_float(row.get("combined_drone_evidence_pct")) or 0.0 for row in ordered), default=0.0),
                "max_candidate_run": max_candidate_run,
                "max_strong_run": max_strong_run,
                "candidate_window_count": sum(1 for flag in candidate if flag),
                "candidate_event_count": _run_count(candidate),
                "alert_window_count": sum(1 for flag in alert if flag),
                "alert_event_count": _run_count(alert),
                "window_count": len(ordered),
                "duration_sec": duration_sec,
            }
        )
    return files


def summarize_group(files: list[dict[str, Any]]) -> dict[str, Any]:
    negatives = [row for row in files if row.get("label") != "drone"]
    positives = [row for row in files if row.get("label") == "drone"]
    negative_hours = sum(float(row.get("duration_sec") or row.get("window_count") or 0) for row in negatives) / 3600.0
    false_candidate_files = sum(1 for row in negatives if row["candidate_any"])
    false_candidate_run2_files = sum(1 for row in negatives if row["candidate_run2"])
    false_candidate_run3_files = sum(1 for row in negatives if row["candidate_run3"])
    false_alert_files = sum(1 for row in negatives if row["alert_any"])
    candidate_any_files = sum(1 for row in files if row["candidate_any"])
    candidate_run2_files = sum(1 for row in files if row["candidate_run2"])
    candidate_run3_files = sum(1 for row in files if row["candidate_run3"])
    strong_run2_files = sum(1 for row in files if row["strong_run2"])
    strong_run3_files = sum(1 for row in files if row["strong_run3"])
    alert_any_files = sum(1 for row in files if row["alert_any"])
    return {
        "files": len(files),
        "candidate_any": candidate_any_files,
        "candidate_run2": candidate_run2_files,
        "candidate_run3": candidate_run3_files,
        "strong_any": sum(1 for row in files if row["strong_any"]),
        "strong_run2": strong_run2_files,
        "strong_run3": strong_run3_files,
        "alert_any": alert_any_files,
        "candidate_any_files": candidate_any_files,
        "candidate_run2_files": candidate_run2_files,
        "candidate_run3_files": candidate_run3_files,
        "strong_run2_files": strong_run2_files,
        "strong_run3_files": strong_run3_files,
        "alert_any_files": alert_any_files,
        "false_candidate_files": false_candidate_files,
        "false_candidate_file_rate": false_candidate_files / len(negatives) if negatives else 0.0,
        "false_candidate_single_spike_files": sum(
            1 for row in negatives if row["candidate_any"] and not row["candidate_run2"]
        ),
        "false_candidate_run2_files": false_candidate_run2_files,
        "false_candidate_run3_files": false_candidate_run3_files,
        "false_alert_files": false_alert_files,
        "false_alert_file_rate": false_alert_files / len(negatives) if negatives else 0.0,
        "false_candidate_events_per_hour": (
            sum(int(row.get("candidate_event_count") or 0) for row in negatives) / negative_hours if negative_hours else 0.0
        ),
        "false_candidate_windows_per_hour": (
            sum(int(row.get("candidate_window_count") or 0) for row in negatives) / negative_hours if negative_hours else 0.0
        ),
        "false_alert_events_per_hour": sum(int(row.get("alert_event_count") or 0) for row in negatives) / negative_hours if negative_hours else 0.0,
        "median_time_to_first_candidate": _median(
            [float(row["time_to_first_candidate_sec"]) for row in positives if row.get("time_to_first_candidate_sec") is not None]
        ),
        "median_max_ml": _median([float(row["max_ml"]) for row in files]),
        "median_max_harmonic": _median([float(row["max_harmonic"]) for row in files]),
        "median_max_combined": _median([float(row["max_combined"]) for row in files]),
    }


def summarize_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = file_metrics(rows)
    summary = {
        "overall": summarize_group(files),
        "by_dataset_id": {},
        "by_label": {},
        "by_distance_category": {},
        "by_source_dataset": {},
        "by_distance_m": {},
        "by_environment": {},
    }
    summary["overall"]["confusion_operator_label"] = dict(
        Counter(str(row.get("operator_label") or "unknown") for row in rows)
    )
    for key, field in (
        ("by_dataset_id", "dataset_id"),
        ("by_label", "label"),
        ("by_distance_category", "distance_category"),
        ("by_source_dataset", "source_dataset"),
        ("by_distance_m", "distance_m"),
        ("by_environment", "environment"),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in files:
            grouped[str(row.get(field) or "unknown")].append(row)
        summary[key] = {name: summarize_group(items) for name, items in sorted(grouped.items())}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SkyEar offline benchmark report CSV.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_report(load_report(args.report))
    body = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
