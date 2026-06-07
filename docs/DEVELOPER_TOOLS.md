# Developer Tools

The operator flow uses:

```bash
skyear server
skyear station
skyear dashboard  # manual snapshot/admin/debug
skyear live       # smooth tactical map
```

Developer shortcuts:

```bash
skyear dev debug-wav runtime/recordings/<session_id>/chunk_0000.wav
skyear dev benchmark --dataset svanstrom --window-sec 1.0 --hf --output-dir reports/benchmark_run
skyear dev simulate
skyear dev simulate fiber-grid --targets 2 --post-realtime --step-sec 0.5 --assert-tracks
```

Legacy scripts remain available for compatibility, including:

- `skyear-stream-local-dataset`
- `skyear-stream-hf-dataset`
- `skyear-build-manifest`
- `skyear-eval-manifest`
- `skyear-run-benchmarks`
- `skyear-simulate-geo-events`
- `skyear-simulate-fiber-grid`
- `skyear-simulate-array-audio`
- `skyear-debug-harmonic-wav`

Fiber-grid deployment simulation:

```bash
skyear server
skyear live
skyear-simulate-fiber-grid \
  --server http://127.0.0.1:8080/events \
  --targets 2 \
  --post-realtime \
  --step-sec 0.5 \
  --station-failure A4,B2 \
  --assert-tracks
```

The simulator posts synthetic `AcousticEvent` and `StationHeartbeat` JSON only. It does not require real audio. The default layout has three passive lines: A at 800m, B at 600m, and C at 400m from the control point.

Use `--post-realtime --step-sec 0.5` when you want to watch movement live. The `/live` tactical map updates in-place without a Streamlit rerun. Without `--post-realtime`, the simulator posts all steps quickly and the live map will mostly show the final state.

The `/live` page is the tactical screen: it shows fusion level, station counts, nearest ETA, latest crossed line, track list, station line overlays, optional bearing sectors, optional track estimates, optional station coverage, and pause/follow controls.

Use schematic mode for offline work:

```text
http://127.0.0.1:8080/live?mode=schematic
```

Use geo mode when `live_map.tile_url` is configured, or when online tiles are explicitly allowed:

```text
http://127.0.0.1:8080/live?mode=geo&lat=<LAT>&lon=<LON>&zoom=13
```

Place a simulation around an operator-chosen control point:

```bash
skyear-simulate-fiber-grid \
  --control-lat <LAT> \
  --control-lon <LON> \
  --target-heading-deg 180 \
  --targets 2 \
  --post-realtime \
  --step-sec 0.5 \
  --assert-tracks
```

`skyear dashboard` is intentionally a manual Streamlit snapshot view. Use **Refresh snapshot** when you want a new admin/debug snapshot. For continuous track movement, ETA, line crossing, and map monitoring, use `skyear live` or open `http://127.0.0.1:8080/live`.

Run normal checks:

```bash
python -m compileall station server dashboard shared tools tests
pytest -q
```
