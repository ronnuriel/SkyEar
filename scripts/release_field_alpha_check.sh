#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MAP_SERVER_PID=""
MAP_BASE_URL="${SKYEAR_BASE_URL:-http://127.0.0.1:18080}"

cleanup() {
  if [[ -n "$MAP_SERVER_PID" ]]; then
    kill "$MAP_SERVER_PID" >/dev/null 2>&1 || true
    wait "$MAP_SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[1/9] Checking clean worktree"
git -C "$ROOT_DIR" status --short
if [[ -n "$(git -C "$ROOT_DIR" status --short)" ]]; then
  echo "Release check requires a clean worktree. Commit or stash changes first." >&2
  exit 1
fi

echo "[2/9] Checking no tracked raw audio"
python -m tools.release_checks no-raw-audio --repo-root "$ROOT_DIR"

echo "[3/9] Checking data/datasets is ignored"
python -m tools.release_checks check-ignore data/datasets/example.wav --repo-root "$ROOT_DIR"

echo "[4/9] Installing build tool"
python -m pip install build

echo "[5/9] Building package"
python -m build

echo "[6/9] Running pytest"
PYTHONPATH=. pytest -q

echo "[7/9] Running release smoke"
bash scripts/release_smoke_test.sh

echo "[8/9] Running map smoke"
if ! curl -fsS "${MAP_BASE_URL%/}/health" >/dev/null 2>&1; then
  python -m uvicorn server.api:app --host 127.0.0.1 --port "${MAP_BASE_URL##*:}" >/tmp/skyear_map_smoke_server.log 2>&1 &
  MAP_SERVER_PID="$!"
  for _ in $(seq 1 30); do
    if curl -fsS "${MAP_BASE_URL%/}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
fi
SKYEAR_BASE_URL="$MAP_BASE_URL" bash scripts/map_smoke_test.sh

echo "[9/9] Running dataset hub smoke"
bash scripts/dataset_hub_smoke_test.sh

echo "Field Alpha release check OK"
