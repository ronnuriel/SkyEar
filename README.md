# SkyEar

SkyEar is a passive acoustic monitoring system for Field Alpha drone-audio testing.

A station listens locally, extracts acoustic evidence, optionally runs a Hugging Face audio classifier, and sends compact JSON events to a central server. The server fuses recent station evidence into tracks. The dashboard shows station health, local decisions, tracks, fusion level, and passive map cues.

SkyEar does not send raw audio to the server. Local raw recording is optional and stays on the station computer.

## Safety Scope

SkyEar is passive warning and engineering evaluation only.

- No jamming
- No interception
- No targeting
- No laser control
- No weapon or countermeasure integration
- PTZ/gimbal support is camera-only visual confirmation
- Field Alpha is an engineering dry-run, not operational deployment

## Quick Start: Field Alpha

Install from the Field Alpha tag:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "skyear[all] @ git+https://github.com/ronnuriel/SkyEar.git@v0.1.0-field-alpha"
```

Copy configs and check audio devices:

```bash
skyear-copy-configs configs
skyear-station --list-devices
```

Run the system in three terminals:

```bash
skyear-server --host 0.0.0.0 --port 8080
```

```bash
skyear-station --config configs/config_station.yaml
```

```bash
skyear-dashboard
```

Open the dashboard:

```text
http://localhost:8501
```

Optional, on the station computer, run the local monitor:

```bash
skyear-local-monitor -- --state runtime/stations/station_001_latest.json --history runtime/stations/station_001_history.jsonl
```

That is the Field Alpha operator flow:

```text
server -> station -> dashboard
optional: local monitor
```

The separate spectrum app is not part of the operator flow.

## What You Should See

The dashboard should show:

- Station health: online, stale, or offline
- Local station status and operator label
- HF/ML drone probability, harmonic evidence, and combined evidence
- Fusion level: `LEVEL 0` to `LEVEL 3`
- Active tracks, with stations grouped by likely shared source
- Passive map cues and bearing sectors when station coordinates/bearings exist
- Operator action text such as observe, take cover, or all clear

The local monitor should show:

- Waveform preview
- Spectrum and harmonic lines
- Spectrogram
- RMS/peak/clipping warnings
- HF label and error state
- Harmonic, ML, and combined evidence bars
- Persistence counters
- f0 and bearing/beam fields when available

## How The Pipeline Works

1. `skyear-station` captures audio from one microphone or a synchronized mic array.
2. The station computes harmonic rotor evidence, f0 stability, RMS/peak, optional beamforming, and optional HF/ML probability.
3. The station writes local JSON snapshots under `runtime/stations/`.
4. The local monitor reads those files directly, so it still works if the server is down.
5. The station posts compact `AcousticEvent` JSON to `/events`.
6. The server stores latest station events and heartbeats.
7. Track fusion groups nearby or matching station detections into active tracks.
8. The dashboard reads `/fusion`, `/stations/*`, `/alerts`, and map endpoints.

HF/ML is advisory. It can support a candidate when acoustic evidence exists, but ML alone must not trigger a public warning.

## Install Options

Install the latest main branch:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "skyear[all] @ git+https://github.com/ronnuriel/SkyEar.git@main"
```

Install from a local checkout for development:

```bash
git clone https://github.com/ronnuriel/SkyEar.git
cd SkyEar
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

After reconnecting over SSH, activate the venv again:

```bash
source .venv/bin/activate
```

Console commands such as `skyear-server`, `skyear-station`, and `skyear-dashboard` exist only inside the active venv unless SkyEar was installed system-wide.

If you build locally:

```bash
python -m pip install build
python -m build
ls dist/
pip install dist/skyear-0.1.0-py3-none-any.whl
```

Release wheels and source archives are attached to GitHub Releases. They are not committed to the repository because `dist/` is ignored.

## Configure One Station

Start by copying configs:

```bash
skyear-copy-configs configs
```

Edit `configs/config_station.yaml`:

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

List devices:

```bash
skyear-station --list-devices
```

Run the station:

```bash
skyear-station --config configs/config_station.yaml
```

## Decision Profiles

SkyEar has two local decision profiles:

- `conservative`: production-safe default. It is stricter about ML probability and persistence, and is designed to reduce false alerts.
- `field_debug`: one-microphone field-test profile. It makes intermediate states more visible, such as `ACOUSTIC DRONE WATCH`, `WEAK LOCAL CANDIDATE`, `LOCAL DRONE CANDIDATE`, and `STRONG LOCAL CANDIDATE`.

`field_debug` is useful when a real flight produces strong harmonic evidence but HF stays below the conservative `0.90` ML threshold. It still blocks single-microphone alert by default.

Example:

```yaml
detection:
  profile: field_debug
  hf_watch_threshold: 0.50
  hf_candidate_threshold: 0.70
  hf_strong_threshold: 0.85
  ml_positive_threshold: 0.90
  single_mic_candidate_run_required: 2
  single_mic_strong_run_required: 3
  allow_single_mic_alert: false
```

For one-mic field tests:

- Harmonic high + HF low becomes `ACOUSTIC HARMONIC SOURCE` or `NON-DRONE HARMONIC`.
- HF above watch threshold + stable harmonic becomes `ACOUSTIC DRONE WATCH`.
- HF above candidate threshold for repeated windows + stable harmonic becomes `LOCAL DRONE CANDIDATE`.
- HF above strong threshold for repeated windows + stable harmonic becomes `STRONG LOCAL CANDIDATE`.
- Harmonic-only evidence must not become alert on one microphone.

For an 8-channel array, start from:

```text
configs/config_station_array_8ch.yaml
```

That config includes explicit `mic_positions_m`, beamforming settings, HF cadence, and local raw recording settings.

## Add Another Station

Copy a config:

```bash
cp configs/config_station.yaml configs/config_station_roof.yaml
```

Change:

- `station.station_id`
- `station.name`
- `station.latitude` / `station.longitude`
- `audio.device_id`
- `local_monitor.state_path`
- `local_monitor.history_path`

Run it:

```bash
skyear-station --config configs/config_station_roof.yaml
```

Run its local monitor:

```bash
skyear-local-monitor -- --state runtime/stations/station_roof_latest.json --history runtime/stations/station_roof_history.jsonl
```

From a source checkout, this helper reads monitor paths from the config:

```bash
scripts/run_station_monitor.sh configs/config_station_roof.yaml
```

## Running A Station On Another Computer

On the server computer:

```bash
skyear-server --host 0.0.0.0 --port 8080
```

On the station computer:

```bash
skyear-copy-configs configs
cp configs/config_station_remote.yaml configs/my_remote_station.yaml
```

Edit the remote config:

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

Check connectivity:

```bash
skyear-check-server --url http://SERVER_IP:8080
```

Start the station:

```bash
skyear-station --config configs/my_remote_station.yaml
```

If the server is unreachable, the station prints a warning and continues local monitor mode. For LAN, Tailscale/WireGuard, ngrok/cloudflared, and reverse proxy setups, see [docs/NETWORKING.md](docs/NETWORKING.md).

## HF Advisory Model

Install with HF support:

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

Smoke test with live captured audio:

```bash
skyear-station --config configs/config_station.yaml --hf-smoke-test
```

Test one WAV file:

```bash
skyear-hf-test --wav some.wav
```

If HF fails, station logs include the real exception message and continue without crashing.

## Demo Without Hardware

Run the server and dashboard first:

```bash
skyear-server --host 0.0.0.0 --port 8080
skyear-dashboard
```

Post a spatial track scenario:

```bash
skyear-simulate-two-near-one-far \
  --server http://127.0.0.1:8080/events \
  --assert-tracks
```

If the console command is not installed in a source checkout, use:

```bash
PYTHONPATH=. python -m tools.simulate_two_near_one_far \
  --server http://127.0.0.1:8080/events \
  --assert-tracks
```

Run the client demo:

```bash
PYTHONPATH=. python -m tools.simulate_client_demo \
  --server http://127.0.0.1:8080/events \
  --channels 8 \
  --window-sec 1.0 \
  --realtime
```

Expected demo behavior:

- Background returns to `LEVEL 0`
- Motorcycle-like false positive should not become a strong alert
- Two-station drone becomes stronger than single-station drone
- Far unrelated stations are shown as separate local candidates or tracks

## Dataset And Benchmark Tools

Public datasets are useful for engineering benchmarks and model training candidates. They are not operational validation by themselves.

Raw datasets live under `data/datasets/` and are ignored by Git.

List and validate datasets:

```bash
skyear-datasets list
skyear-datasets validate
```

Small Hugging Face smoke download:

```bash
skyear-download-datasets \
  --dataset droneaudioset_hf \
  --hf-config drone-only \
  --split train_001 \
  --max-examples 20
```

SkyEar refuses to materialize large HF datasets unless `--force-large` is passed. Prefer `--metadata-only`, `--streaming-export`, or `--max-examples` first.

Common dataset IDs and aliases:

```text
droneaudioset_hf
drone_audio_detection_samples_hf  aliases: dads_hf, dads
drone_detection_thesis_github     aliases: svanstrom, svanstrom_drone_detection
sara_alemadi_github
acoustic_uav_github
bowony_github
kaggle_yehiel_levi
```

Build and evaluate a manifest:

```bash
skyear-build-manifest \
  --registry data/dataset_registry.yaml \
  --dataset svanstrom \
  --verify-audio \
  --output data/manifests/svanstrom_manifest.csv

skyear-stream-manifest \
  --manifest data/manifests/svanstrom_manifest.csv \
  --config configs/config_station.yaml \
  --mode offline \
  --window-sec 1.0 \
  --save-report reports/svanstrom/eval.csv

skyear-summarize-benchmark \
  --report reports/svanstrom/eval.csv \
  --output reports/svanstrom/summary.json
```

One-command benchmark runner:

```bash
skyear-run-benchmarks \
  --dataset svanstrom \
  --window-sec 1.0 \
  --hf \
  --output-dir reports/benchmark_run
```

## Field Test Sessions

Use field session tools to make real tests reproducible and useful for later evaluation.

Read the protocol first:

```text
docs/FIELD_TEST_PROTOCOL.md
```

Start a session:

```bash
skyear-start-field-session \
  --location "north test field" \
  --station-id station_001 \
  --drone-model DJI_Neo
```

Mark an event:

```bash
skyear-mark-field-event \
  --session field_sessions/<session_id> \
  --label drone \
  --distance-m 50 \
  --drone-model DJI_Neo \
  --bearing-deg 0 \
  --note "hover 30 sec north"
```

Save a manual debug capture:

```bash
skyear-save-debug-capture \
  --seconds 30 \
  --label unknown \
  --note "manual capture"
```

Evaluate the session:

```bash
skyear-eval-field-session \
  --session field_sessions/<session_id> \
  --output-json field_sessions/<session_id>/reports/eval_summary.json
```

New `notes.csv` files use `timestamp` and `bearing_deg`. Older notes with `timestamp_unix` or `ground_truth_bearing_deg` are still supported.

## Map And Passive Geo Cues

Set station coordinates in each station config:

```yaml
station:
  station_id: station_array_8ch_001
  name: Field Array 8ch
  latitude: 32.0853
  longitude: 34.7818
  altitude_m: 20
  heading_offset_deg: 0
  location_label: "north tripod"
```

A single station can show a range-unknown bearing sector. An approximate candidate area requires at least two recent stations with valid passive bearings, or a simulated/known source used for testing.

Map smoke test:

```bash
bash scripts/map_smoke_test.sh
```

## Security For Field Use

Local development can accept unauthenticated events. When exposing the server outside localhost, configure token or HMAC authentication.

Server:

```bash
export SKYEAR_API_TOKEN="change-me"
export SKYEAR_HMAC_SECRET="change-me-too"
skyear-server --host 0.0.0.0 --port 8080
```

Station config:

```yaml
server:
  url: http://SERVER_IP:8080/events
  api_token: change-me
  hmac_secret: change-me-too
```

## Developer / Debug Tools

The Field Alpha operator flow does not use the old spectrum app. Operators should run only:

```text
skyear-server
skyear-station
skyear-dashboard
optional: skyear-local-monitor
```

For developer debugging from a source checkout, the old spectrum app is still available:

```bash
PYTHONPATH=. streamlit run dashboard/station_spectrum_app.py --server.port 8502
```

It is not part of the Field Alpha startup flow and is not installed as a console command.

## Release And Tests

Field Alpha release checklist:

```text
docs/RELEASE_FIELD_ALPHA.md
```

Run the normal checks:

```bash
python -m compileall station server dashboard shared tools tests
pytest -q
bash -n scripts/release_field_alpha_check.sh scripts/run_dataset_benchmarks.sh scripts/release_smoke_test.sh
python -m build
```

Full release preflight:

```bash
bash scripts/release_field_alpha_check.sh
```

## Alert Levels

- `LEVEL 0`: background
- `LEVEL 1`: local or single-station candidate, operator observe
- `LEVEL 2`: network acoustic confirmation candidate or strong local candidate
- `LEVEL 3`: stronger multi-station candidate; still requires human validation before any public warning
