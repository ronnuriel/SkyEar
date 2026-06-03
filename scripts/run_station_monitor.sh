#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/config_station.yaml}"

STATE_PATH="$(
  PYTHONPATH=. python - "$CONFIG_PATH" <<'PY'
import sys
import yaml
from pathlib import Path

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
station_id = cfg.get("station", {}).get("station_id", "station_001")
local_cfg = cfg.get("local_monitor", {})
directory = Path(local_cfg.get("directory", "runtime/stations"))
print(local_cfg.get("state_path") or directory / f"{station_id}_latest.json")
PY
)"

HISTORY_PATH="$(
  PYTHONPATH=. python - "$CONFIG_PATH" <<'PY'
import sys
import yaml
from pathlib import Path

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
station_id = cfg.get("station", {}).get("station_id", "station_001")
local_cfg = cfg.get("local_monitor", {})
directory = Path(local_cfg.get("directory", "runtime/stations"))
print(local_cfg.get("history_path") or directory / f"{station_id}_history.jsonl")
PY
)"

PYTHONPATH=. streamlit run dashboard/local_station_app.py -- \
  --state "$STATE_PATH" \
  --history "$HISTORY_PATH"
