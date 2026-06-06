# Recording

SkyEar records raw audio locally on the station machine. Dashboard and server controls send commands only; raw audio is not uploaded by default.

Privacy note: recording may capture voices. Use only where permitted.

## Commands

```bash
skyear rec start home_test
skyear rec mark hover --note "DJI Neo 20m" --distance-m 20 --drone-model "DJI Neo"
skyear rec stop
skyear rec state
skyear rec summary
```

Legacy commands still work:

```bash
skyear-recording-start --config configs/config_station.yaml --session-name home_test
skyear-recording-mark --config configs/config_station.yaml --label hover --note "DJI Neo 20m"
skyear-recording-stop --config configs/config_station.yaml
```

## Files

Each session is saved under `runtime/recordings/<session_id>/`:

- `chunk_0000.wav`, `chunk_0001.wav`, ...
- `metadata.json`
- `markers.csv`
- `station_config_snapshot.json`

Useful health fields:

- `recording_blocks_written`
- `audio_input_overflow_count`
- `overflow_recent`
- `overflow_timestamps`
- `recording_continuity_ok`
- `detection_blocks_dropped`
- `capture_queue_depth`
- `discontinuities`
- `marker_count`

Summarize latest recording:

```bash
skyear rec summary
```

Build a local recording manifest:

```bash
skyear-build-recording-manifest --root runtime/recordings --output data/manifests/local_recordings_manifest.csv
```
