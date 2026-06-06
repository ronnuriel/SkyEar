# Architecture

SkyEar is intentionally split into station, server, dashboard, shared schema, configs, tools, and runtime output.

```text
microphones -> station capture -> recording writer
                            -> detector pipeline
                            -> local snapshot
                            -> server
                            -> dashboard
```

## Components

- `station/`: audio capture, local recording, harmonic/HF detection, two-mic direction, local snapshots, event posting.
- `server/`: FastAPI API, event store, station heartbeat, recording command queue, fusion, map state.
- `dashboard/`: central operator UI and local monitor UI.
- `shared/`: Pydantic event schemas and auth helpers.
- `configs/`: operator-facing profiles.
- `tools/`: CLI, recording tools, dataset/benchmark/debug/release tools.
- `runtime/`: generated local files; ignored by git.

## Station Flow

The station owns raw audio. It captures microphone blocks, writes local recordings when enabled, runs the detector pipeline, writes a local monitor snapshot, and posts compact JSON events to the server.

HF/ML is advisory. It may support a candidate when acoustic evidence exists, but ML alone must not produce an alert.

Station responsibility modules:

- `station/capture.py`: audio capture facade.
- `station/detection_pipeline.py`: detector/HF/harmonic facade.
- `station/direction_pipeline.py`: beamforming, bearing tracking, and two-mic direction facade.
- `station/recording_integration.py`: recording manager/control facade.
- `station/local_snapshot.py`: local monitor snapshot helpers.
- `station/event_builder.py`: future event assembly home.
- `station/heartbeat_client.py`: future heartbeat/posting home.
- `station/station_agent.py`: orchestration entry point.

## Server Flow

The server stores events and heartbeats, exposes station health, queues recording commands for stations, fuses recent events, and exposes map state.

## Dashboard Flow

The dashboard reads server endpoints for central operation. The local monitor reads `runtime/stations/*` directly so it can work when the server is down.
