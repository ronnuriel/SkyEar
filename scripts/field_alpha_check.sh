#!/usr/bin/env bash
set -u

CONFIG_DIR="${SKYEAR_FIELD_ALPHA_CONFIG_DIR:-/tmp/skyear_field_alpha_configs}"
SERVER_URL="${SKYEAR_FIELD_ALPHA_SERVER_URL:-http://127.0.0.1:8080}"
STATION_CONFIG="$CONFIG_DIR/config_station_array_8ch.yaml"

echo "== SkyEar Field-Alpha preflight =="
echo

echo "Copying packaged configs to: $CONFIG_DIR"
skyear-copy-configs --output "$CONFIG_DIR"
echo

echo "Audio input devices:"
if ! skyear-station --list-devices; then
  echo "[WARN] Could not list audio devices. Check sounddevice/PortAudio installation."
fi
echo

echo "Checking server: $SERVER_URL"
if ! skyear-check-server --url "$SERVER_URL"; then
  echo "[WARN] Server check failed. This is OK for offline local-monitor dry-runs."
fi
echo

echo "HF smoke test:"
python - <<'PY'
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except Exception as exc:
    print(f"[SKIP] HF dependencies are not installed: {type(exc).__name__}: {exc}")
    raise SystemExit(42)
raise SystemExit(0)
PY
hf_deps=$?
if [ "$hf_deps" -eq 0 ]; then
  if ! skyear-station --config "$STATION_CONFIG" --hf-smoke-test; then
    echo "[WARN] HF smoke test failed. Inspect the printed HF error before field collection."
  fi
elif [ "$hf_deps" -ne 42 ]; then
  echo "[WARN] Could not determine HF dependency status."
fi
echo

cat <<EOF
Next commands:

1. Start server:
   skyear-server --host 0.0.0.0 --port 8080

2. Start a field session:
   skyear-start-field-session --station-id station_array_8ch_001 --drone-model DJI_Neo --location "FIELD_LOCATION"

3. Start station:
   skyear-station --config $STATION_CONFIG

4. Start local monitor:
   skyear-local-monitor -- --state runtime/stations/station_array_8ch_001_latest.json --history runtime/stations/station_array_8ch_001_history.jsonl

5. Mark a baseline:
   skyear-mark-field-event --session field_sessions/<session_id> --label background --note "2 min no-drone baseline"

6. Mark a drone run:
   skyear-mark-field-event --session field_sessions/<session_id> --label drone --distance-m 50 --drone-model DJI_Neo --bearing-deg 0 --note "hover 30 sec north"
EOF
