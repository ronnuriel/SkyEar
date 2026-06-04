#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-reports/sim}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
mkdir -p "$OUT_DIR"

TXT_SUMMARY="$OUT_DIR/hard_test_summary.txt"
CSV_SUMMARY="$OUT_DIR/hard_test_summary.csv"
JSON_SUMMARY="$OUT_DIR/hard_test_summary.json"
: >"$TXT_SUMMARY"
printf "case,median_error_deg,p90_error_deg,detection_rate,median_confidence,reliable_rate\n" >"$CSV_SUMMARY"

log_line() {
  echo "$*" | tee -a "$TXT_SUMMARY"
}

printf "%-24s %16s %14s %15s %18s %14s\n" \
  "case" "median_error" "p90_error" "detection_rate" "median_confidence" "reliable_rate" | tee -a "$TXT_SUMMARY"

run_case() {
  local name="$1"
  local max_median="${2:-}"
  shift 2
  local wav="$OUT_DIR/${name}.wav"
  local truth="$OUT_DIR/${name}_truth.csv"
  local eval="$OUT_DIR/${name}_eval.csv"

  PYTHONPATH=. python -m tools.simulate_array_audio \
    --profile field_8ch_r0_35m \
    --sample-rate 48000 \
    --duration-sec 6 \
    --bearing-start-deg 300 \
    --bearing-end-deg 60 \
    --source-type harmonic_drone \
    --f0 1200 \
    --output "$wav" \
    --truth "$truth" \
    "$@" >/dev/null

  PYTHONPATH=. python -m tools.eval_array_audio \
    --wav "$wav" \
    --truth "$truth" \
    --config configs/config_station_array_8ch.yaml \
    --window-sec 1.0 \
    --output "$eval" >/dev/null

  PYTHONPATH=. python - "$name" "$eval" "$max_median" "$CSV_SUMMARY" <<'PY' | tee -a "$TXT_SUMMARY"
import csv
import math
import statistics
import sys

name, path, max_median, summary_csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
errors = []
confidences = []
reliable = 0
detections = 0
rows = 0
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        rows += 1
        if row.get("predicted_bearing_deg"):
            detections += 1
        if row.get("bearing_error_deg"):
            errors.append(float(row["bearing_error_deg"]))
        if row.get("beam_confidence"):
            confidences.append(float(row["beam_confidence"]))
        if str(row.get("bearing_reliable")).lower() == "true":
            reliable += 1
if not errors:
    raise SystemExit(f"{name}: no bearing estimates")
median_error = statistics.median(errors)
p90_error = sorted(errors)[min(len(errors) - 1, int(math.ceil(len(errors) * 0.9)) - 1)]
detection_rate = detections / max(1, rows)
reliable_rate = reliable / max(1, rows)
median_confidence = statistics.median(confidences) if confidences else float("nan")
print(f"{name:<24} {median_error:16.2f} {p90_error:14.2f} {detection_rate:15.2f} {median_confidence:18.2f} {reliable_rate:14.2f}")
with open(summary_csv, "a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "case",
            "median_error_deg",
            "p90_error_deg",
            "detection_rate",
            "median_confidence",
            "reliable_rate",
        ],
    )
    writer.writerow(
        {
            "case": name,
            "median_error_deg": f"{median_error:.6f}",
            "p90_error_deg": f"{p90_error:.6f}",
            "detection_rate": f"{detection_rate:.6f}",
            "median_confidence": f"{median_confidence:.6f}",
            "reliable_rate": f"{reliable_rate:.6f}",
        }
    )
if detection_rate < 0.50:
    raise SystemExit(f"{name}: detection rate collapsed: {detection_rate:.2f}")
if max_median:
    threshold = float(max_median)
    if median_error > threshold:
        raise SystemExit(f"{name}: median error {median_error:.2f} > {threshold:.2f}")
PY
}

run_case clean 5 --snr-db 25
run_case snr_0db 15 --snr-db 0
run_case snr_minus_5db "" --snr-db -5
run_case position_jitter_5cm "" --snr-db 10 --mic-position-jitter-cm 5
run_case gain_jitter_3db "" --snr-db 10 --mic-gain-jitter-db 3
run_case delay_jitter_100us "" --snr-db 10 --channel-delay-jitter-us 100
run_case one_dropped_channel "" --snr-db 10 --drop-channel 3
run_case multipath "" --snr-db 10 --reflection-count 3 --reflection-delay-ms 5,12,25 --reflection-gain-db=-6,-10,-15 --reflection-bearing-offset-deg 40
run_case harmonic_interferer "" --snr-db 10 --interferer-type harmonic --interferer-bearing-deg 180 --interferer-f0 1000 --interferer-snr-db 0

PYTHONPATH=. python - "$CSV_SUMMARY" "$JSON_SUMMARY" <<'PY'
import csv
import json
import sys

csv_path, json_path = sys.argv[1], sys.argv[2]
rows = []
with open(csv_path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        converted = {"case": row["case"]}
        for key, value in row.items():
            if key == "case":
                continue
            converted[key] = float(value)
        rows.append(converted)
with open(json_path, "w", encoding="utf-8") as handle:
    json.dump({"cases": rows}, handle, indent=2, sort_keys=True)
PY

log_line "wrote $TXT_SUMMARY"
log_line "wrote $CSV_SUMMARY"
log_line "wrote $JSON_SUMMARY"
