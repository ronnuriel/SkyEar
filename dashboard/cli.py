from __future__ import annotations

import sys
from pathlib import Path


def _run_streamlit(script_name: str, extra_args: list[str] | None = None) -> None:
    from streamlit.web import cli as streamlit_cli

    script_path = Path(__file__).with_name(script_name)
    if not script_path.exists():
        print(f"SkyEar dashboard app not found: {script_path}", file=sys.stderr)
        raise SystemExit(2)
    sys.argv = ["streamlit", "run", str(script_path), *(extra_args or sys.argv[1:])]
    raise SystemExit(streamlit_cli.main())


def main_dashboard() -> None:
    _run_streamlit("app.py")


def main_spectrum() -> None:
    _run_streamlit("station_spectrum_app.py")


def main_local_monitor() -> None:
    _run_streamlit("local_station_app.py")
