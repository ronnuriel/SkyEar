from __future__ import annotations

import argparse
from pathlib import Path

from tools.field_session import FIELD_LABELS, append_field_note


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a ground-truth note to a SkyEar field session.")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--label", choices=sorted(FIELD_LABELS), required=True)
    parser.add_argument("--distance-m", type=float)
    parser.add_argument("--drone-model", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--bearing-deg", type=float)
    parser.add_argument("--ground-truth-bearing-deg", type=float)
    parser.add_argument("--maneuver", default="")
    parser.add_argument("--station-id", default="")
    parser.add_argument("--timestamp-unix", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = append_field_note(
        session=args.session,
        label=args.label,
        distance_m=args.distance_m,
        drone_model=args.drone_model,
        note=args.note,
        bearing_deg=args.bearing_deg,
        ground_truth_bearing_deg=args.ground_truth_bearing_deg,
        maneuver=args.maneuver,
        station_id=args.station_id,
        timestamp_unix=args.timestamp_unix,
    )
    print(f"marked {row['label']} at {row['iso_time']} in {args.session}")


if __name__ == "__main__":
    main()
