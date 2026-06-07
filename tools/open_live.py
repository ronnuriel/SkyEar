from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def live_url_from_config(config_path: str | Path = "configs/config_station.yaml", *, explicit_url: str | None = None) -> str:
    if explicit_url:
        url = str(explicit_url)
    else:
        cfg = _load_config(config_path)
        url = str((cfg.get("server", {}) or {}).get("url") or "http://127.0.0.1:8080/live")
    if url.endswith("/events"):
        url = url[: -len("/events")] + "/live"
    elif not url.endswith("/live"):
        url = url.rstrip("/") + "/live"
    return url


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the SkyEar live tactical map.")
    parser.add_argument("--config", default="configs/config_station.yaml")
    parser.add_argument("--url", help="Explicit live URL, e.g. http://127.0.0.1:8080/live")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    url = live_url_from_config(args.config, explicit_url=args.url)
    print(url)
    if args.print_only:
        return 0
    try:
        opened = webbrowser.open(url)
    except Exception:
        opened = False
    if not opened:
        print(f"Open this URL in your browser: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
