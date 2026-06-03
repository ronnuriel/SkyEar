#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${SKYEAR_BASE_URL:-http://127.0.0.1:8080}"
EVENTS_URL="${BASE_URL%/}/events"

curl -fsS "${BASE_URL%/}/health" >/dev/null

if command -v skyear-simulate-geo-events >/dev/null 2>&1; then
  SIM_CMD=(skyear-simulate-geo-events)
else
  SIM_CMD=(python -m tools.simulate_geo_events)
fi

"${SIM_CMD[@]}" \
  --server "$EVENTS_URL" \
  --station-a-lat 31.9955 \
  --station-a-lon 34.0000 \
  --station-a-bearing 0 \
  --station-b-lat 32.0000 \
  --station-b-lon 34.0053 \
  --station-b-bearing 270 \
  --heartbeat

MAP_STATE="$(curl -fsS "${BASE_URL%/}/map/state")"
export MAP_STATE

python - <<'PY'
import json
import os
import sys

state = json.loads(os.environ["MAP_STATE"])
stations = state.get("stations") or []
cues = state.get("bearing_cues") or []
estimates = state.get("geo_estimates") or []

if len(stations) < 2:
    raise SystemExit(f"expected at least 2 stations, got {len(stations)}")
if len(cues) < 2:
    raise SystemExit(f"expected at least 2 bearing cues, got {len(cues)}")
if not estimates:
    raise SystemExit("expected at least 1 geo estimate")

estimate = estimates[0]
if estimate.get("estimate_type") not in {"bearing_intersection", "multi_station_area"}:
    raise SystemExit(f"unexpected estimate_type: {estimate.get('estimate_type')}")

print(
    "map smoke OK:",
    f"stations={len(stations)}",
    f"bearing_cues={len(cues)}",
    f"estimate_type={estimate.get('estimate_type')}",
    f"lat={estimate.get('latitude')}",
    f"lon={estimate.get('longitude')}",
    f"radius_m={estimate.get('radius_m')}",
    f"confidence={estimate.get('confidence')}",
    f"geometry={estimate.get('bearing_geometry_quality')}",
)
PY
