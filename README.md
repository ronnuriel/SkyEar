# SkyEar

SkyEar is a passive acoustic monitoring system for drone-audio experiments. A station listens locally, records optional raw audio locally, extracts harmonic and HF/ML evidence, and sends compact JSON events to a central server and dashboard.

Raw audio is not uploaded to the server by default. Recordings stay under `runtime/recordings/` on the station machine.

## Safety Scope

SkyEar is passive warning and engineering evaluation only.

- No jamming
- No interception
- No targeting
- No laser control
- No weapon or countermeasure integration
- PTZ/gimbal support is camera-only visual confirmation

## Quick Start

Install from a local checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
skyear-copy-configs configs
skyear setup audio
```

Run the system in three terminals:

```bash
skyear server
```

```bash
skyear station
```

```bash
skyear dashboard
```

Optional local station monitor:

```bash
skyear monitor
```

The default config is `configs/config_station.yaml`. Override it with:

```bash
skyear --config configs/my_station.yaml station
SKYEAR_CONFIG=configs/my_station.yaml skyear station
```

## Basic Flow

```text
microphones -> station capture -> local recording writer
                            -> detector pipeline
                            -> local snapshot
                            -> server
                            -> dashboard
```

Project components:

- `station/`: audio capture, recording, harmonic/HF detection, two-mic direction, local snapshots, event posting.
- `server/`: FastAPI API, event store, station heartbeat, recording command queue, fusion, map state.
- `dashboard/`: central operator UI and local monitor UI.
- `shared/`: Pydantic event schemas and auth helpers.
- `configs/`: operator-facing profiles.
- `tools/`: grouped CLI, recording, dataset, benchmark, debug, and release tools.
- `runtime/`: generated local files; ignored by git.

## Operator Commands

Recording:

```bash
skyear rec start home_test
skyear rec mark hover --note "test 20m"
skyear rec stop
skyear rec summary
```

Checks:

```bash
skyear setup audio --dry-run
skyear check audio --diagnostic-sec 20
skyear check two-mic --tracked
skyear check server
skyear check hf
```

Release:

```bash
skyear release preflight
skyear release tag v0.2.0-field-alpha --push
```

The older `skyear-*` commands still work for backward compatibility.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Operator Guide](docs/OPERATOR_GUIDE.md)
- [Station Config](docs/STATION_CONFIG.md)
- [Recording](docs/RECORDING.md)
- [Two-Mic Direction](docs/TWO_MIC_DIRECTION.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Developer Tools](docs/DEVELOPER_TOOLS.md)
- [Datasets And Benchmarks](docs/DATASETS_AND_BENCHMARKS.md)
- [Release](docs/RELEASE.md)

## Alert Levels

- `LEVEL 0`: background
- `LEVEL 1`: local or single-station candidate, operator observe
- `LEVEL 2`: network acoustic confirmation candidate or strong local candidate
- `LEVEL 3`: stronger multi-station candidate; still requires human validation before any public warning
