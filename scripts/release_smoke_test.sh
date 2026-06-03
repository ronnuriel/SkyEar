#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/skyear_release_smoke.XXXXXX")"
VENV_DIR="$TMP_DIR/venv"

python -m build "$ROOT_DIR"
python -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install "$ROOT_DIR"/dist/skyear-*.whl

mkdir -p "$TMP_DIR/empty"
pushd "$TMP_DIR/empty" >/dev/null
if skyear-station 2>"$TMP_DIR/station_missing_config.err"; then
  echo "Expected skyear-station without default config to fail cleanly" >&2
  exit 1
fi
grep -q "Default config not found. Run: skyear-copy-configs ./configs or pass --config PATH" "$TMP_DIR/station_missing_config.err"
popd >/dev/null

skyear-copy-configs --output "$TMP_DIR/configs"
test -f "$TMP_DIR/configs/config_station.yaml"
test -f "$TMP_DIR/configs/config_station_remote.yaml"

skyear-copy-configs "$TMP_DIR/configs2"
test -f "$TMP_DIR/configs2/config_station.yaml"
test -f "$TMP_DIR/configs2/config_station_remote.yaml"

skyear-check-server --help >/dev/null
skyear-simulate-geo-events --help >/dev/null
echo "release smoke OK: $TMP_DIR"
