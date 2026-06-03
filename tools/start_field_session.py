from __future__ import annotations

import argparse
from pathlib import Path

from tools.field_session import create_field_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a SkyEar field test session folder.")
    parser.add_argument("--root", type=Path, default=Path("field_sessions"))
    parser.add_argument("--session-id")
    parser.add_argument("--location", default="")
    parser.add_argument("--station-id", action="append", dest="station_ids", default=[])
    parser.add_argument("--weather", default="")
    parser.add_argument("--wind-estimate", default="")
    parser.add_argument("--drone-model", default="")
    parser.add_argument("--operator-notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = create_field_session(
        root=args.root,
        session_id=args.session_id,
        location=args.location,
        station_ids=args.station_ids,
        weather=args.weather,
        wind_estimate=args.wind_estimate,
        drone_model=args.drone_model,
        operator_notes=args.operator_notes,
    )
    print(session_dir)


if __name__ == "__main__":
    main()
