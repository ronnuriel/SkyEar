#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-reports/sim}"
WAV_PATH="$OUT_DIR/moving_source_8ch.wav"
TRUTH_PATH="$OUT_DIR/moving_source_truth.csv"
EVAL_PATH="$OUT_DIR/beam_eval.csv"
MAX_MEDIAN_ERROR_DEG="${MAX_MEDIAN_ERROR_DEG:-15}"

PYTHONPATH=. python -m tools.simulate_array_audio \
  --profile field_8ch_r0_35m \
  --sample-rate 48000 \
  --duration-sec 6 \
  --bearing-start-deg 300 \
  --bearing-end-deg 60 \
  --source-type harmonic_drone \
  --f0 1200 \
  --snr-db 20 \
  --output "$WAV_PATH" \
  --truth "$TRUTH_PATH"

PYTHONPATH=. python -m tools.eval_array_audio \
  --wav "$WAV_PATH" \
  --truth "$TRUTH_PATH" \
  --config configs/config_station_array_8ch.yaml \
  --window-sec 1.0 \
  --output "$EVAL_PATH"

PYTHONPATH=. python - "$EVAL_PATH" "$MAX_MEDIAN_ERROR_DEG" <<'PY'
import csv
import statistics
import sys

path = sys.argv[1]
threshold = float(sys.argv[2])
errors = []
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("bearing_error_deg"):
            errors.append(float(row["bearing_error_deg"]))
if not errors:
    raise SystemExit("No bearing estimates were produced")
median_error = statistics.median(errors)
print(f"synthetic array median_error_deg={median_error:.2f} threshold={threshold:.2f}")
if median_error > threshold:
    raise SystemExit(f"Median bearing error too high: {median_error:.2f} > {threshold:.2f}")
PY

if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
  PYTHONPATH=. python -m tools.simulate_moving_geo \
    --server http://127.0.0.1:8080/events \
    --station-a-lat 32.10350 \
    --station-a-lon 34.80800 \
    --station-b-lat 32.10420 \
    --station-b-lon 34.80920 \
    --path-start-lat 32.10520 \
    --path-start-lon 34.80830 \
    --path-end-lat 32.10540 \
    --path-end-lon 34.81030 \
    --steps 5 \
    --interval-sec 0.05
  curl -fsS http://127.0.0.1:8080/map/state >/dev/null
else
  echo "SkyEar server is not running on 127.0.0.1:8080; skipped moving geo map smoke."
fi
