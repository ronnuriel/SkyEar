from __future__ import annotations

import argparse
from pathlib import Path

from tools.field_session import save_debug_capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a lightweight local monitor debug capture.")
    parser.add_argument("--state", type=Path, default=Path("runtime/stations/station_001_latest.json"))
    parser.add_argument("--history", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("field_debug_captures"))
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--label", default="unknown")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_dir = save_debug_capture(
        state_path=args.state,
        history_path=args.history,
        output_root=args.output_root,
        seconds=args.seconds,
        label=args.label,
        note=args.note,
    )
    print(capture_dir)


if __name__ == "__main__":
    main()
