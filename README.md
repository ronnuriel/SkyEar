# SkyEar

Passive acoustic early-warning network for drone-like rotor signatures.

## Safety Scope

- Passive detection and warning only
- No jamming
- No interception
- No blinding
- No targeting
- No laser control
- No active countermeasure

## Architecture

Station nodes capture audio, run local detection, and send `AcousticEvent` JSON to the server. The server stores recent events, fuses station evidence, and exposes API endpoints to the operator dashboard.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Terminal 1:

```bash
PYTHONPATH=. uvicorn server.api:app --reload --host 0.0.0.0 --port 8080
```

Terminal 2:

```bash
PYTHONPATH=. python -m station.station_agent --config configs/config_station.yaml
```

Terminal 3:

```bash
PYTHONPATH=. streamlit run dashboard/app.py
```

Terminal 4, optional dedicated spectrum page:

```bash
PYTHONPATH=. streamlit run dashboard/station_spectrum_app.py --server.port 8502
```

## Operator Dashboard

The main dashboard is a tactical overview. It shows compact station cards, fusion level, status, confidence, harmonic score, best f0, channel agreement, strongest channel, RMS, and duration.

Use **Open Spectrum** on any station card to inspect the dedicated spectrum app. The detailed page shows the full latest spectrum, harmonic markers, spectrogram, and per-channel evidence without making the main dashboard flicker.

## Client Demo

Run a complete passive demo with simulated stations:

```bash
PYTHONPATH=. python -m tools.simulate_client_demo \
  --server http://127.0.0.1:8080/events \
  --channels 8 \
  --realtime
```

Expected phases:

- `background`: all stations remain background.
- `motorcycle_false_positive_test`: motorcycle-like audio should not produce ALERT.
- `two_station_drone`: two simulated stations detect drone-like harmonic evidence and fusion rises.
- `single_station_drone`: one station remains affected, so fusion drops from the multi-station case.
- `all_clear`: stations clear back to background.

## HF Advisory Model

Optional Hugging Face audio classification support is advisory only. It can raise confidence when rotor-harmonic evidence exists, but it cannot trigger ALERT by itself.

Install optional dependencies:

```bash
pip install -r requirements_hf.txt
```

Enable in `configs/config_station.yaml`:

```yaml
hf:
  enabled: true
  model_id: preszzz/drone-audio-detection-05-17-trial-0
  run_every_n_windows: 2
  threshold: 0.70
  fallback_drone_label_idx: 1
```

## False Positive Tuning

SkyEar prioritizes reliability over sensitivity. Speech, music, motorcycles, and fans can contain harmonics, so the detector requires persistent rotor-like evidence before escalating. `SUSPECT` can appear quickly, but `DRONE_LIKE` and `ALERT` require stable f0, multi-channel agreement, or advisory model support, and ALERT still requires rotor-harmonic evidence plus minimum duration.

Tune these in `configs/config_station.yaml`:

```yaml
stability:
  enabled: true
  history_windows: 4
  max_f0_std_hz: 80
  min_score_windows: 3
```

## Tests

```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. python -m compileall station server dashboard shared tools tests
```

Alert levels:

- LEVEL 0: background
- LEVEL 1: suspect, operator only
- LEVEL 2: drone-like, security/operator
- LEVEL 3: confirmed by fusion, public-warning candidate only after validation
