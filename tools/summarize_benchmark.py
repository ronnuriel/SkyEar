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
        first_candidate = next((_float(row.get("window_idx")) for row in ordered if _is_candidate(row)), None)
        files.append(
            {
                "file_path": file_path,
                "dataset_id": ordered[0].get("dataset_id") or "unknown",
                "label": ordered[0].get("label") or "unknown",
                "distance_category": ordered[0].get("distance_category") or "unknown",
                "candidate_any": any(candidate),
                "candidate_run2": _max_run(candidate) >= 2,
                "candidate_run3": _max_run(candidate) >= 3,
                "strong_any": any(strong),
                "strong_run2": _max_run(strong) >= 2,
                "strong_run3": _max_run(strong) >= 3,
                "alert_any": any(alert),
                "time_to_first_candidate_sec": first_candidate,
                "max_ml": max((_float(row.get("ml_drone_pct") or row.get("hf_p_drone")) or 0.0 for row in ordered), default=0.0),
                "max_harmonic": max(
                    (_float(row.get("harmonic_evidence_pct_smoothed") or row.get("harmonic_evidence_pct")) or 0.0 for row in ordered),
                    default=0.0,
                ),
                "max_combined": max((_float(row.get("combined_drone_evidence_pct")) or 0.0 for row in ordered), default=0.0),
                "window_count": len(ordered),
            }
        )
    return files


def summarize_group(files: list[dict[str, Any]]) -> dict[str, Any]:
    negatives = [row for row in files if row.get("label") != "drone"]
    positives = [row for row in files if row.get("label") == "drone"]
    negative_hours = sum(float(row.get("window_count") or 0) for row in negatives) / 3600.0
    return {
        "files": len(files),
        "candidate_any": sum(1 for row in files if row["candidate_any"]),
        "candidate_run2": sum(1 for row in files if row["candidate_run2"]),
        "candidate_run3": sum(1 for row in files if row["candidate_run3"]),
        "strong_any": sum(1 for row in files if row["strong_any"]),
        "strong_run2": sum(1 for row in files if row["strong_run2"]),
        "strong_run3": sum(1 for row in files if row["strong_run3"]),
        "alert_any": sum(1 for row in files if row["alert_any"]),
        "false_candidate_events_per_hour": (
            sum(1 for row in negatives if row["candidate_any"]) / negative_hours if negative_hours else 0.0
        ),
        "false_candidate_windows_per_hour": (
            sum(int(row.get("window_count") or 0) for row in negatives if row["candidate_any"]) / negative_hours
            if negative_hours
            else 0.0
        ),
        "false_alert_events_per_hour": sum(1 for row in negatives if row["alert_any"]) / negative_hours if negative_hours else 0.0,
        "median_time_to_first_candidate": _median(
            [float(row["time_to_first_candidate_sec"]) for row in positives if row.get("time_to_first_candidate_sec") is not None]
        ),
        "median_max_ml": _median([float(row["max_ml"]) for row in files]),
        "median_max_harmonic": _median([float(row["max_harmonic"]) for row in files]),
        "median_max_combined": _median([float(row["max_combined"]) for row in files]),
    }


def summarize_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = file_metrics(rows)
    summary = {"overall": summarize_group(files), "by_dataset_id": {}, "by_label": {}, "by_distance_category": {}}
    summary["overall"]["confusion_operator_label"] = dict(
        Counter(str(row.get("operator_label") or "unknown") for row in rows)
    )
    for key, field in (("by_dataset_id", "dataset_id"), ("by_label", "label"), ("by_distance_category", "distance_category")):
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
