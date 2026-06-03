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
CANDIDATE_STATUSES = {"drone_like", "alert"}
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
        max_run = _max_run(candidate_flags)
        first = ordered[0]
        results.append(
            {
                "file_path": file_path,
                "label": first.get("label"),
                "source_dataset": first.get("source_dataset") or "unknown",
                "split_group": first.get("split_group") or first.get("source_dataset") or "unknown",
                "window_count": len(ordered),
                "candidate_any": any(candidate_flags),
                "candidate_run2": max_run >= 2,
                "candidate_run3": max_run >= 3,
                "max_candidate_run": max_run,
                "truth_drone": _truth_is_drone(first),
            }
        )
    return results


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = file_level_metrics(rows)
    positives = [row for row in files if row["truth_drone"]]
    negatives = [row for row in files if not row["truth_drone"]]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return sum(1 for item in items if item[key]) / len(items)

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
        source_pos = [item for item in source_files if item["truth_drone"]]
        source_neg = [item for item in source_files if not item["truth_drone"]]
        source_holdout[source] = {
            "files": len(source_files),
            "drone_files": len(source_pos),
            "non_drone_files": len(source_neg),
            "candidate_any_recall": rate(source_pos, "candidate_any"),
            "candidate_run2_recall": rate(source_pos, "candidate_run2"),
            "candidate_run3_recall": rate(source_pos, "candidate_run3"),
            "false_positive_files": false_by_source[source],
            "false_positive_rate": rate(source_neg, "candidate_any"),
        }

    return {
        "window_level": {
            "windows": len(rows),
            "candidate_windows": sum(1 for row in rows if _is_candidate(row)),
        },
        "file_level": {
            "files": len(files),
            "drone_files": len(positives),
            "non_drone_files": len(negatives),
            "candidate_any_recall": rate(positives, "candidate_any"),
            "candidate_run2_recall": rate(positives, "candidate_run2"),
            "candidate_run3_recall": rate(positives, "candidate_run3"),
            "false_positive_files": sum(1 for item in negatives if item["candidate_any"]),
            "false_positive_rate": rate(negatives, "candidate_any"),
        },
        "false_positives_per_source": dict(sorted(false_by_source.items())),
        "source_holdout": source_holdout,
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
