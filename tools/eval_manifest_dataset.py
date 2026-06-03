from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CANDIDATE_LABELS = {
    "ml_drone_candidate",
    "local_drone_candidate",
    "strong_local_candidate",
    "drone_like",
    "alert",
}
STRONG_LABELS = {"strong_local_candidate", "drone_like", "alert"}
CANDIDATE_STATUSES = {"drone_like", "alert"}
STRONG_STATUSES = {"drone_like", "alert"}
ALERT_LABELS = {"alert"}
ALERT_STATUSES = {"alert"}
DRONE_LABELS = {"drone"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth_is_drone(row: dict[str, Any]) -> bool:
    return str(row.get("label") or "").lower() in DRONE_LABELS


def _is_candidate(row: dict[str, Any]) -> bool:
    label = str(row.get("operator_label") or "").lower()
    status = str(row.get("status") or "").lower()
    return label in CANDIDATE_LABELS or status in CANDIDATE_STATUSES


def _is_strong(row: dict[str, Any]) -> bool:
    label = str(row.get("operator_label") or "").lower()
    status = str(row.get("status") or "").lower()
    return label in STRONG_LABELS or status in STRONG_STATUSES


def _is_alert(row: dict[str, Any]) -> bool:
    label = str(row.get("operator_label") or "").lower()
    status = str(row.get("status") or "").lower()
    return label in ALERT_LABELS or status in ALERT_STATUSES


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_float(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        for key in keys:
            value = _float_or_none(row.get(key))
            if value is not None:
                values.append(value)
                break
    return max(values) if values else None


def _row_time_sec(row: dict[str, Any], first_timestamp: float | None = None) -> float | None:
    for key in ("window_start_sec", "time_sec", "offset_sec"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    timestamp = _float_or_none(row.get("timestamp_unix"))
    if timestamp is not None and first_timestamp is not None:
        return max(0.0, timestamp - first_timestamp)
    window_idx = _float_or_none(row.get("window_idx"))
    window_sec = _float_or_none(row.get("window_sec"))
    if window_idx is not None and window_sec is not None:
        return window_idx * window_sec
    return None


def _time_to_first(rows: list[dict[str, Any]], predicate) -> float | None:
    timestamps = [_float_or_none(row.get("timestamp_unix")) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    first_timestamp = min(timestamps) if timestamps else None
    for row in rows:
        if predicate(row):
            return _row_time_sec(row, first_timestamp)
    return None


def _max_int(rows: list[dict[str, Any]], key: str) -> int:
    values = []
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            values.append(int(value))
    return max(values) if values else 0


def _file_duration_sec(rows: list[dict[str, Any]]) -> float:
    durations = [_float_or_none(row.get("duration_sec")) for row in rows]
    durations = [value for value in durations if value is not None and value > 0]
    if durations:
        return max(durations)
    window_secs = [_float_or_none(row.get("window_sec")) for row in rows]
    window_secs = [value for value in window_secs if value is not None and value > 0]
    if window_secs:
        return len(rows) * max(window_secs)
    times = [_row_time_sec(row) for row in rows]
    times = [value for value in times if value is not None]
    if times:
        return max(times) - min(times)
    return 0.0


def _max_run(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _manifest_by_path(manifest_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("file_path")): row for row in manifest_rows if row.get("file_path")}


def merge_predictions_with_manifest(
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest = _manifest_by_path(manifest_rows)
    merged = []
    for row in prediction_rows:
        file_path = str(row.get("file_path") or "")
        base = dict(manifest.get(file_path, {}))
        base.update(row)
        if not base.get("source_dataset"):
            base["source_dataset"] = "unknown"
        if not base.get("split_group"):
            base["split_group"] = base.get("source_dataset") or "unknown"
        merged.append(base)
    return merged


def file_level_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("file_path") or row.get("file_name") or "unknown")].append(row)

    results = []
    for file_path, file_rows in grouped.items():
        ordered = sorted(file_rows, key=lambda item: int(float(item.get("window_idx") or 0)))
        candidate_flags = [_is_candidate(row) for row in ordered]
        strong_flags = [_is_strong(row) for row in ordered]
        alert_flags = [_is_alert(row) for row in ordered]
        max_candidate_run = _max_run(candidate_flags)
        max_strong_run_flags = _max_run(strong_flags)
        first = ordered[0]
        max_candidate_run_field = _max_int(ordered, "candidate_run")
        max_strong_run_field = _max_int(ordered, "strong_run")
        results.append(
            {
                "file_path": file_path,
                "label": first.get("label"),
                "source_dataset": first.get("source_dataset") or "unknown",
                "split_group": first.get("split_group") or first.get("source_dataset") or "unknown",
                "distance_m": first.get("distance_m"),
                "distance_category": first.get("distance_category"),
                "environment": first.get("environment"),
                "window_count": len(ordered),
                "candidate_any": any(candidate_flags),
                "candidate_run2": max_candidate_run >= 2,
                "candidate_run3": max_candidate_run >= 3,
                "strong_any": any(strong_flags),
                "strong_run2": max_strong_run_flags >= 2,
                "strong_run3": max_strong_run_flags >= 3,
                "alert_any": any(alert_flags),
                "time_to_first_candidate_sec": _time_to_first(ordered, _is_candidate),
                "time_to_first_strong_sec": _time_to_first(ordered, _is_strong),
                "max_ml": _max_float(ordered, "ml_drone_pct", "hf_p_drone"),
                "max_harmonic": _max_float(ordered, "harmonic_evidence_pct_smoothed", "harmonic_evidence_pct", "harmonic_score"),
                "max_combined": _max_float(ordered, "combined_drone_evidence_pct"),
                "max_candidate_run": max(max_candidate_run, max_candidate_run_field),
                "max_strong_run": max(max_strong_run_flags, max_strong_run_field),
                "duration_sec": _file_duration_sec(ordered),
                "truth_drone": _truth_is_drone(first),
            }
        )
    return results


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _group_key(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " | ".join(str(item.get(key) or "unknown") for key in keys)


def _summary_for_files(files: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in files if row["truth_drone"]]
    negatives = [row for row in files if not row["truth_drone"]]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return sum(1 for item in items if item.get(key)) / len(items)

    negative_hours = sum(float(item.get("duration_sec") or 0.0) for item in negatives) / 3600.0
    positive_times = [
        float(item["time_to_first_candidate_sec"])
        for item in positives
        if item.get("time_to_first_candidate_sec") is not None
    ]
    return {
        "files": len(files),
        "drone_files": len(positives),
        "non_drone_files": len(negatives),
        "false_candidate_any_rate": rate(negatives, "candidate_any"),
        "false_candidate_run2_rate": rate(negatives, "candidate_run2"),
        "false_candidate_run3_rate": rate(negatives, "candidate_run3"),
        "false_strong_run2_rate": rate(negatives, "strong_run2"),
        "false_alert_rate": rate(negatives, "alert_any"),
        "false_candidates_per_hour": (
            sum(1 for item in negatives if item.get("candidate_any")) / negative_hours if negative_hours > 0 else 0.0
        ),
        "false_alerts_per_hour": (
            sum(1 for item in negatives if item.get("alert_any")) / negative_hours if negative_hours > 0 else 0.0
        ),
        "recall_candidate_run2": rate(positives, "candidate_run2"),
        "recall_candidate_run3": rate(positives, "candidate_run3"),
        "median_time_to_first_candidate": _median(positive_times),
        "candidate_any_recall": rate(positives, "candidate_any"),
        "candidate_run2_recall": rate(positives, "candidate_run2"),
        "candidate_run3_recall": rate(positives, "candidate_run3"),
        "false_positive_files": sum(1 for item in negatives if item.get("candidate_any")),
        "false_positive_rate": rate(negatives, "candidate_any"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = file_level_metrics(rows)
    false_by_source: dict[str, int] = defaultdict(int)
    totals_by_source: dict[str, int] = defaultdict(int)
    source_holdout: dict[str, dict[str, Any]] = {}
    for item in files:
        source = str(item.get("source_dataset") or "unknown")
        totals_by_source[source] += 1
        if not item["truth_drone"] and item["candidate_any"]:
            false_by_source[source] += 1

    for source in sorted(totals_by_source):
        source_files = [item for item in files if item["source_dataset"] == source]
        source_holdout[source] = _summary_for_files(source_files)

    grouped: dict[str, dict[str, Any]] = {}
    for keys in (
        ("source_dataset",),
        ("label",),
        ("distance_m",),
        ("distance_category",),
        ("environment",),
        ("source_dataset", "label"),
        ("source_dataset", "distance_category"),
    ):
        bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in files:
            bucket[_group_key(item, keys)].append(item)
        grouped["/".join(keys)] = {key: _summary_for_files(items) for key, items in sorted(bucket.items())}

    return {
        "window_level": {
            "windows": len(rows),
            "candidate_windows": sum(1 for row in rows if _is_candidate(row)),
            "strong_windows": sum(1 for row in rows if _is_strong(row)),
            "alert_windows": sum(1 for row in rows if _is_alert(row)),
        },
        "file_level": _summary_for_files(files),
        "false_positives_per_source": dict(sorted(false_by_source.items())),
        "source_holdout": source_holdout,
        "grouped": grouped,
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SkyEar predictions against an audio manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="CSV/JSONL report from dataset streaming.")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_predictions_with_manifest(load_rows(args.manifest), load_rows(args.predictions))
    summary = summarize(rows)
    body = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
