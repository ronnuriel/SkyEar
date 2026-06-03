from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.field_session import field_session_summary, load_prediction_rows, read_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SkyEar field session.")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        default=[],
        help="Optional CSV/JSONL station report or history file. Can be passed more than once.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def evaluate_session(session: Path, predictions: list[Path] | None = None) -> dict:
    notes = read_csv_rows(session / "notes.csv")
    rows = load_prediction_rows(session, predictions)
    return field_session_summary(notes=notes, prediction_rows=rows)


def main() -> None:
    args = parse_args()
    summary = evaluate_session(args.session, args.predictions)
    body = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
