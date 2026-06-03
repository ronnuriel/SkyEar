#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/skyear_dataset_hub_smoke.XXXXXX")"
CONFIG="$TMP_DIR/config.yaml"
DATASET_ROOT="$TMP_DIR/fixtures/svanstrom"
MANIFEST="$TMP_DIR/manifest.csv"
REPORT="$TMP_DIR/report.csv"
SUMMARY="$TMP_DIR/summary.json"
SINGLE_REPORT="$TMP_DIR/single_report.csv"
SPLITS_DIR="$TMP_DIR/splits"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

run_cli() {
  local command_name="$1"
  local module_name="$2"
  shift 2
  if command -v "$command_name" >/dev/null 2>&1; then
    "$command_name" "$@"
  else
    python -m "$module_name" "$@"
  fi
}

run_cli skyear-datasets tools.dataset_registry validate
run_cli skyear-download-datasets tools.download_datasets \
  --dataset droneaudioset_hf \
  --hf-config drone-only \
  --split train_001 \
  --max-examples 2 \
  --dry-run

python - "$DATASET_ROOT" "$CONFIG" <<'PY'
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

root = Path(sys.argv[1])
config = Path(sys.argv[2])
sample_rate = 16000
t = np.arange(sample_rate, dtype=np.float32) / sample_rate
tone = (0.05 * np.sin(2 * math.pi * 700 * t)).astype(np.float32)
silence = np.zeros(sample_rate, dtype=np.float32)
for rel, audio in {
    "Close/drone/DRONE_001.wav": tone,
    "Distant/background/BACKGROUND_001.wav": silence,
}.items():
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, audio)

config.write_text(
    yaml.safe_dump(
        {
            "station": {"station_id": "dataset_hub_smoke"},
            "audio": {"channels": 1, "sample_rate": sample_rate, "window_sec": 1.0},
        }
    ),
    encoding="utf-8",
)
PY

run_cli skyear-build-manifest tools.build_audio_manifest --root "$DATASET_ROOT" --output-csv "$MANIFEST" --skip-audio-hash
run_cli skyear-stream-manifest tools.stream_manifest_dataset --manifest "$MANIFEST" --config "$CONFIG" --mode offline --max-windows 1 --save-report "$REPORT"
run_cli skyear-summarize-benchmark tools.summarize_benchmark --report "$REPORT" --output "$SUMMARY" >/dev/null
run_cli skyear-eval-audio tools.eval_audio --wav "$DATASET_ROOT/Close/drone/DRONE_001.wav" --config "$CONFIG" --label drone --save-report "$SINGLE_REPORT"
run_cli skyear-build-training-splits tools.build_training_splits --manifest "$MANIFEST" --output-dir "$SPLITS_DIR"

test -s "$MANIFEST"
test -s "$REPORT"
test -s "$SUMMARY"
test -s "$SINGLE_REPORT"
test -s "$SPLITS_DIR/train_manifest.csv"

echo "dataset hub smoke OK: $TMP_DIR"
