# Drone Acoustic Network

Passive acoustic early-warning network for drone-like rotor signatures.

Safety scope:
- Passive detection and warning only
- No jamming
- No interception
- No blinding
- No targeting
- No active countermeasure

## Architecture

Station nodes capture audio, run local detection, and send event JSON to the server.
The server fuses events from multiple stations, tracks alert levels, and exposes data to the dashboard.

## Run MVP locally

python -m station.station_agent --list-devices

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Terminal 1:

```bash
uvicorn server.api:app --reload --host 0.0.0.0 --port 8080
```

Terminal 2:

```bash
python -m station.station_agent --config configs/config_station.yaml
```

Terminal 3:

```bash
streamlit run dashboard/app.py
```

Alert levels:
- LEVEL 0: background
- LEVEL 1: suspect, operator only
- LEVEL 2: drone-like, security/operator
- LEVEL 3: confirmed by fusion, public-warning candidate only after validation
