from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


RAW_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def tracked_raw_audio_files(repo_root: str | Path = ".") -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_root),
        check=True,
        text=True,
        capture_output=True,
    )
    files = []
    for line in result.stdout.splitlines():
        if Path(line).suffix.lower() in RAW_AUDIO_SUFFIXES:
            files.append(line)
    return files


def git_check_ignore(path: str | Path, repo_root: str | Path = ".") -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=str(repo_root),
        text=True,
    )
    return result.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SkyEar release safety checks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    raw = subparsers.add_parser("no-raw-audio")
    raw.add_argument("--repo-root", type=Path, default=Path("."))
    ignored = subparsers.add_parser("check-ignore")
    ignored.add_argument("path")
    ignored.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "no-raw-audio":
        files = tracked_raw_audio_files(args.repo_root)
        if files:
            print("Tracked raw audio files are not allowed in a Field Alpha release:")
            for path in files:
                print(path)
            raise SystemExit(1)
        print("no tracked raw audio files")
        return
    if args.command == "check-ignore":
        if not git_check_ignore(args.path, args.repo_root):
            raise SystemExit(f"expected ignored path is not ignored: {args.path}")
        print(f"ignored: {args.path}")


if __name__ == "__main__":
    main()
