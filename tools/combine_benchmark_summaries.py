from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COMBINED_FIELDNAMES = [
    "dataset_id",
    "files",
    "drone_candidate_run2_rate",
    "drone_candidate_run3_rate",
    "drone_strong_run2_rate",
    "background_false_run2_rate",
    "helicopter_false_run2_rate",
    "false_alert_file_rate",
    "median_time_to_first_candidate",
    "notes",
]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _rate(numerator: Any, denominator: Any) -> float:
    denominator_f = _float(denominator)
    if denominator_f <= 0.0:
        return 0.0
    return _float(numerator) / denominator_f


def _label_group(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return dict(((summary.get("by_label") or {}).get(label) or {}))


def combined_row(dataset_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    overall = dict(summary.get("overall") or {})
    drone = _label_group(summary, "drone")
    background = _label_group(summary, "background")
    helicopter = _label_group(summary, "helicopter")
    notes = []
    if not drone:
        notes.append("no drone label group")
    if not background:
        notes.append("no background label group")
    return {
        "dataset_id": dataset_id,
        "files": int(_float(overall.get("files"), 0.0)),
        "drone_candidate_run2_rate": _rate(drone.get("candidate_run2_files"), drone.get("files")),
        "drone_candidate_run3_rate": _rate(drone.get("candidate_run3_files"), drone.get("files")),
        "drone_strong_run2_rate": _rate(drone.get("strong_run2_files"), drone.get("files")),
        "background_false_run2_rate": _rate(background.get("false_candidate_run2_files"), background.get("files")),
        "helicopter_false_run2_rate": _rate(helicopter.get("false_candidate_run2_files"), helicopter.get("files")),
        "false_alert_file_rate": _float(overall.get("false_alert_file_rate"), 0.0),
        "median_time_to_first_candidate": overall.get("median_time_to_first_candidate"),
        "notes": "; ".join(notes),
    }


def find_summary_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*/summary.json") if path.is_file())


def combine_summaries(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in find_summary_files(input_dir):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(combined_row(path.parent.name, summary))
    return rows


def write_combined_outputs(rows: list[dict[str, Any]], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMBINED_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine SkyEar benchmark summary.json files.")
    parser.add_argument("--input-dir", type=Path, default=Path("reports"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = combine_summaries(args.input_dir)
    output_json = args.output_json or args.input_dir / "combined_summary.json"
    output_csv = args.output_csv or args.input_dir / "combined_summary.csv"
    write_combined_outputs(rows, output_json, output_csv)
    print(f"combined_summaries={len(rows)} json={output_json} csv={output_csv}")


if __name__ == "__main__":
    main()
