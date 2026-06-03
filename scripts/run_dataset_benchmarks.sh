#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if command -v skyear-run-benchmarks >/dev/null 2>&1; then
  exec skyear-run-benchmarks "$@"
fi

exec python -m tools.run_dataset_benchmarks "$@"
