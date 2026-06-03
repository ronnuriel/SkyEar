from __future__ import annotations

import argparse
from pathlib import Path

from tools.stream_manifest_dataset import run_single_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one audio file with the SkyEar offline station detector.")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config_station.yaml"))
    parser.add_argument("--label", default="unknown")
    parser.add_argument("--save-report", type=Path)
    parser.add_argument("--window-sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_single_wav(
        wav=args.wav,
        config=args.config,
        label=args.label,
        save_report=args.save_report,
        window_sec=args.window_sec,
    )


if __name__ == "__main__":
    main()
