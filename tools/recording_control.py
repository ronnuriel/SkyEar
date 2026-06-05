from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
import yaml


def _load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _base_url(config: dict[str, Any]) -> str:
    rec = config.get("recording", {}) or {}
    host = str(rec.get("local_control_host", "127.0.0.1"))
    port = int(rec.get("local_control_port", 8765))
    return f"http://{host}:{port}"


def _print_state(payload: dict[str, Any]) -> None:
    state = payload.get("state") if isinstance(payload.get("state"), dict) else payload
    print(json.dumps(payload, indent=2, sort_keys=True))
    if isinstance(state, dict):
        print(
            "summary:",
            f"ok={payload.get('ok', True)}",
            f"recording={state.get('recording')}",
            f"session_dir={state.get('session_dir')}",
            f"marker_count={state.get('marker_count')}",
            f"wav_files_count={len(state.get('wav_files') or [])}",
            f"last_error={state.get('last_error')}",
        )
        wavs = state.get("wav_files") or []
        if wavs:
            print("wav_files:")
            for item in wavs:
                path = item.get("wav_path") if isinstance(item, dict) else item
                print(f"  {path}")


def _post(config_path: str, endpoint: str, payload: dict[str, Any]) -> None:
    cfg = _load_config(config_path)
    response = requests.post(f"{_base_url(cfg)}{endpoint}", json=payload, timeout=5)
    response.raise_for_status()
    _print_state(response.json())


def _get(config_path: str, endpoint: str) -> None:
    cfg = _load_config(config_path)
    response = requests.get(f"{_base_url(cfg)}{endpoint}", timeout=5)
    response.raise_for_status()
    _print_state(response.json())


def start_main() -> None:
    parser = argparse.ArgumentParser(description="Start local SkyEar station recording.")
    parser.add_argument("--config", default="configs/config_station.yaml")
    parser.add_argument("--session-name", default="session")
    parser.add_argument("--label", default=None)
    parser.add_argument("--note", default=None)
    args = parser.parse_args()
    _post(args.config, "/recording/start", {"session_name": args.session_name, "label": args.label, "note": args.note})


def stop_main() -> None:
    parser = argparse.ArgumentParser(description="Stop local SkyEar station recording.")
    parser.add_argument("--config", default="configs/config_station.yaml")
    args = parser.parse_args()
    _post(args.config, "/recording/stop", {})


def state_main() -> None:
    parser = argparse.ArgumentParser(description="Show local SkyEar station recording state.")
    parser.add_argument("--config", default="configs/config_station.yaml")
    args = parser.parse_args()
    _get(args.config, "/recording/state")


def mark_main() -> None:
    parser = argparse.ArgumentParser(description="Mark an event in the active local SkyEar recording.")
    parser.add_argument("--config", default="configs/config_station.yaml")
    parser.add_argument("--label", required=True)
    parser.add_argument("--note", default=None)
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--bearing-deg", type=float, default=None)
    parser.add_argument("--drone-model", default=None)
    args = parser.parse_args()
    _post(
        args.config,
        "/recording/mark",
        {
            "label": args.label,
            "note": args.note,
            "distance_m": args.distance_m,
            "bearing_deg": args.bearing_deg,
            "drone_model": args.drone_model,
            "source": "manual",
        },
    )
