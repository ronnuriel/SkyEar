# SkyEar

SkyEar is a passive acoustic monitoring system for drone-like rotor evidence. A station listens locally, runs signal processing and optional ML/Hugging Face inference, writes a local live monitor snapshot, and can send compact JSON events to a central server for multi-station fusion.

## Safety Scope

SkyEar is passive detection and operator warning only.

- No jamming
- No interception
- No blinding
- No targeting
- No laser control
- No weapon or active countermeasure integration
- PTZ/gimbal support is camera-only visual confirmation

## How The Pipeline Works

1. `skyear-station` captures audio from one station microphone or mic array.
2. The station computes harmonic rotor evidence, f0 stability, RMS/peak, optional beam/bearing, and optional HF/ML drone probability.
3. The station writes a local latest snapshot JSON and compact JSONL history under `runtime/stations/`.
4. The local monitor reads that JSON directly, so it keeps working even if the central server is down.
5. If a server URL is configured, the station posts an `AcousticEvent` JSON to `/events`.
6. The central server stores latest station events and heartbeats, clusters station evidence into tracks, and exposes `/fusion`.
7. The operator dashboard shows station health, local decisions, tracks, fusion level, bearing cues, and recommended operator action.

Raw audio is not sent to the server. Optional raw recording writes local WAV snippets around local candidates only when enabled.

## Install From GitHub

Install the base station/server/dashboard tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "skyear @ git+https://github.com/ronnuriel/SkyEar.git@main"
```

Install with Hugging Face and dataset tools:

```bash
pip install "skyear[all] @ git+https://github.com/ronnuriel/SkyEar.git@main"
```

Copy example configs into the current directory:

```bash
skyear-copy-configs configs
```

You can also copy them to a temporary or deployment directory:

```bash
skyear-copy-configs /tmp/skyear_configs
```

Console commands such as `skyear-station` and `skyear-server` are available inside the active virtual environment unless you installed SkyEar system-wide. After reconnecting over SSH, activate the venv again:

```bash
source /tmp/skyear_git_install_test/bin/activate
```

## Install From A Local Checkout

```bash
git clone https://github.com/ronnuriel/SkyEar.git
cd SkyEar
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For HF/dataset evaluation:

```bash
pip install -e ".[all,dev]"
```

Legacy requirements files are also kept:

```bash
pip install -r requirements.txt
pip install -r requirements_hf.txt
```

## Quick Start

Terminal 1, central server:

```bash
skyear-server --host 0.0.0.0 --port 8080
```

Terminal 2, one live station:

```bash
skyear-station --config configs/config_station.yaml
```

Terminal 3, local per-station monitor:

```bash
skyear-local-monitor -- --state runtime/stations/station_001_latest.json --history runtime/stations/station_001_history.jsonl
```

If you are running from a source checkout, the helper script reads the monitor paths from the config:

```bash
scripts/run_station_monitor.sh configs/config_station.yaml
```

Terminal 4, central operator dashboard:

```bash
skyear-dashboard
```

Optional dedicated spectrum app:

```bash
skyear-spectrum -- --server.port 8502
```

If running from source without installing console scripts, prefix the old commands with `PYTHONPATH=.`:

```bash
PYTHONPATH=. uvicorn server.api:app --host 0.0.0.0 --port 8080
PYTHONPATH=. python -m station.station_agent --config configs/config_station.yaml
PYTHONPATH=. streamlit run dashboard/app.py
```

## Station Setup

List audio devices:

```bash
skyear-station --list-devices
```

Edit the station config:

```yaml
station:
  station_id: station_001
  name: North Roof
  latitude: 32.0853
  longitude: 34.7818

audio:
  device_id: 0
  sample_rate: 44100
  channels: 1
  window_sec: 1.0

server:
  url: http://127.0.0.1:8080/events

local_monitor:
  enabled: true
  state_path: runtime/stations/station_001_latest.json
  history_path: runtime/stations/station_001_history.jsonl
```

Start it:

```bash
skyear-station --config configs/config_station.yaml
```

## Add A New Station

1. Copy a config:

```bash
cp configs/config_station.yaml configs/config_station_roof.yaml
```

2. Change:

- `station.station_id`
- `station.name`
- `station.latitude` / `station.longitude`
- `audio.device_id`
- `local_monitor.state_path`
- `local_monitor.history_path`

3. Start the station:

```bash
skyear-station --config configs/config_station_roof.yaml
```

4. Start its local monitor:

```bash
skyear-local-monitor -- --state runtime/stations/station_roof_latest.json --history runtime/stations/station_roof_history.jsonl
```

In a source checkout you can also use:

```bash
scripts/run_station_monitor.sh configs/config_station_roof.yaml
```

For an 8-channel synchronized array, start from:

```bash
configs/config_station_array_8ch.yaml
```

That config includes `mic_positions_m`, beamforming, HF cadence, and local raw recording settings.

## Running Station On Another Computer

Run the central server on one computer:

```bash
skyear-server --host 0.0.0.0 --port 8080
```

Find that computer's LAN/VPN IP, then on the station computer copy the remote template:

```bash
cp configs/config_station_remote.yaml configs/my_remote_station.yaml
```

Edit:

```yaml
station:
  station_id: remote_station_001
  name: Remote Station 001

audio:
  device_id: 0

server:
  url: http://SERVER_IP:8080/events

local_monitor:
  enabled: true
```

Check connectivity from the station computer:

```bash
skyear-check-server --url http://SERVER_IP:8080
```

Start the station:

```bash
skyear-station --config configs/my_remote_station.yaml
```

If the central server is unreachable, the station prints a warning and still continues local monitor mode. For same-LAN, VPN, temporary tunnel, and reverse proxy deployments, see [docs/NETWORKING.md](docs/NETWORKING.md).

## Local Monitor

The station writes:

- `runtime/stations/<station_id>_latest.json`
- `runtime/stations/<station_id>_history.jsonl`

The local monitor displays:

- Waveform preview
- Spectrum and harmonic lines
- Spectrogram
- HF drone probability and label
- Harmonic evidence
- Combined drone evidence
- Persistence counters
- f0, RMS, peak/clipping warning
- Beam/bearing panel
- Server send and heartbeat errors

It does not require the central server. If `/events` is down, the local station monitor still updates from the JSON file.

## Central Dashboard

```bash
skyear-dashboard
```

The central dashboard shows:

- All latest stations
- Station health and heartbeat age
- Operator label and decision reason
- Fusion level
- Active tracks
- Bearing cue rows
- Operator action: `observe`, `take cover`, or `all clear`

The dedicated spectrum app can be opened from each station card.

## Optional HF Advisory Model

HF is advisory only. It can support a local candidate when acoustic evidence exists, but it must not trigger ALERT alone.

Install:

```bash
pip install "skyear[hf] @ git+https://github.com/ronnuriel/SkyEar.git@main"
```

Enable in config:

```yaml
hf:
  enabled: true
  model_id: preszzz/drone-audio-detection-05-17-trial-0
  run_every_n_windows: 2
  threshold: 0.70
  fallback_drone_label_idx: 1
```

Smoke test on live captured audio:

```bash
skyear-station --config configs/config_station.yaml --hf-smoke-test
```

Test on a WAV file:

```bash
skyear-hf-test --wav some.wav
```

## Demo And Dataset Evaluation

Simulated client demo:

```bash
PYTHONPATH=. python -m tools.simulate_client_demo \
  --server http://127.0.0.1:8080/events \
  --channels 8 \
  --realtime
```

Stream local WAV/FLAC/MP3 dataset files:

```bash
skyear-stream-local-dataset \
  --root data/datasets/svanstrom_drone_detection \
  --label-filter drone \
  --distance-filter Distant \
  --station-id svan_distant_drone \
  --channels 1 \
  --window-sec 1.0 \
  --hf \
  --realtime \
  --server http://127.0.0.1:8080/events
```

Build a manifest:

```bash
skyear-build-manifest \
  --root data/datasets/svanstrom_drone_detection \
  --output-csv reports/manifest.csv \
  --output-jsonl reports/manifest.jsonl
```

Evaluate a saved station report:

```bash
skyear-eval-manifest \
  --manifest reports/manifest.csv \
  --predictions reports/svanstrom_policy/drone.csv \
  --output-json reports/eval_summary.json
```

## Dataset Hub And Offline Benchmarks

Raw public datasets live under `data/datasets/` and are ignored by Git. Track only registry, manifests, reports, and code. Public datasets are useful for engineering benchmarks and training candidates, but they are not operational validation by themselves.

List and validate registered sources:

```bash
skyear-datasets list
skyear-datasets validate
```

Download or prepare a registered source:

```bash
skyear-download-datasets --dataset drone_audio_detection_samples_hf
```

Build a registry-backed manifest:

```bash
skyear-build-manifest \
  --registry data/dataset_registry.yaml \
  --dataset drone_audio_detection_samples_hf \
  --verify-audio \
  --output data/manifests/all_sources_manifest.csv
```

Run the station detector offline over a manifest:

```bash
skyear-stream-manifest \
  --manifest data/manifests/all_sources_manifest.csv \
  --config configs/config_station.yaml \
  --mode offline \
  --window-sec 1.0 \
  --save-report reports/manifest_eval.csv
```

Evaluate one WAV:

```bash
skyear-eval-audio \
  --wav path/to/file.wav \
  --config configs/config_station.yaml \
  --label unknown \
  --save-report reports/single_wav.csv
```

Summarize benchmark output and build leakage-safe splits:

```bash
skyear-summarize-benchmark \
  --report reports/manifest_eval.csv \
  --output reports/benchmark_summary.json

skyear-build-training-splits \
  --manifest data/manifests/all_sources_manifest.csv \
  --output-dir data/manifests
```

## Field Test Sessions

Use the field-session tools for engineering dry-runs and labeled data collection. Start with the full checklist in [docs/FIELD_TEST_PROTOCOL.md](docs/FIELD_TEST_PROTOCOL.md).

```bash
skyear-start-field-session \
  --location "north test field" \
  --station-id station_001 \
  --drone-model DJI_Neo

skyear-mark-field-event \
  --session field_sessions/<session_id> \
  --label drone \
  --distance-m 50 \
  --drone-model DJI_Neo \
  --bearing-deg 0 \
  --note "hover 30 sec north"

skyear-save-debug-capture \
  --seconds 30 \
  --label unknown \
  --note "manual capture"

skyear-eval-field-session \
  --session field_sessions/<session_id> \
  --output-json field_sessions/<session_id>/reports/eval_summary.json
```

New `notes.csv` files use `timestamp` and `bearing_deg`. Older notes with `timestamp_unix` or `ground_truth_bearing_deg` are still supported by the evaluator.

## Map View / Passive Geo Cues

The dashboard includes a passive map section for station locations, health, bearing sectors, and approximate multi-station acoustic estimates. These are estimated acoustic areas and bearing cues, not targeting-grade positions.

Set station coordinates in each station config:

```yaml
station:
  id: station_array_8ch_001
  station_id: station_array_8ch_001
  name: Field Array 8ch
  latitude: 32.0853
  longitude: 34.7818
  altitude_m: 20
  heading_offset_deg: 0
  location_label: "north tripod"
```

Run server, station, and dashboard:

```bash
skyear-server --host 0.0.0.0 --port 8080
skyear-station --config configs/config_station_array_8ch.yaml
skyear-dashboard
```

A single station can only show a range-unknown bearing sector. An approximate candidate location point requires at least two recent stations with valid passive bearings, or an explicitly simulated/known position.

Test the map without microphones:

```bash
skyear-simulate-geo-events \
  --server http://127.0.0.1:8080/events \
  --station-a-lat 31.9955 --station-a-lon 34.0000 --station-a-bearing 0 \
  --station-b-lat 32.0000 --station-b-lon 34.0053 --station-b-bearing 270
```

## Security

By default, local development accepts unauthenticated station events. For field use, set one or both environment variables on the server:

```bash
export SKYEAR_API_TOKEN="change-me"
export SKYEAR_HMAC_SECRET="change-me-too"
skyear-server --host 0.0.0.0 --port 8080
```

Then configure stations:

```yaml
server:
  url: http://SERVER_IP:8080/events
  api_token: change-me
  hmac_secret: change-me-too
```

## Build A Release Package

Build a wheel and source distribution:

```bash
python -m pip install --upgrade build
python -m build
```

Artifacts are written to `dist/`.

Install the wheel locally:

```bash
pip install dist/skyear-0.1.0-py3-none-any.whl
```

Create a GitHub release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Upload the files from `dist/` to the GitHub release, or install directly from GitHub:

```bash
pip install "skyear[all] @ git+https://github.com/ronnuriel/SkyEar.git@v0.1.0"
```

## Tests

```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. python -m compileall station server dashboard shared tools tests
```

## Alert Levels

- `LEVEL 0`: background
- `LEVEL 1`: local/single-station candidate, operator observe
- `LEVEL 2`: network acoustic confirmation candidate or strong local candidate
- `LEVEL 3`: stronger multi-station candidate; still requires operational validation before any public warning
