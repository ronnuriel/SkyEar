from __future__ import annotations

import argparse
import shutil
from importlib import resources
from pathlib import Path


CONFIG_FILES = (
    "config_station.yaml",
    "config_station_2.yaml",
    "config_station_array_8ch.yaml",
    "config_station_remote.yaml",
)


def copy_configs(destination: Path, overwrite: bool = False) -> list[Path]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    package_files = resources.files("configs")
    for name in CONFIG_FILES:
        target = destination / name
        if target.exists() and not overwrite:
            continue
        with resources.as_file(package_files / name) as source:
            shutil.copyfile(source, target)
        copied.append(target)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy packaged SkyEar station config templates.")
    parser.add_argument("destination", nargs="?", default="configs", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    copied = copy_configs(args.destination, overwrite=args.overwrite)
    if copied:
        for path in copied:
            print(path)
    else:
        print(f"No configs copied; files already exist in {args.destination}. Use --overwrite to replace them.")


if __name__ == "__main__":
    main()
