# Developer Tools

The operator flow uses:

```bash
skyear server
skyear station
skyear dashboard
```

Developer shortcuts:

```bash
skyear dev debug-wav runtime/recordings/<session_id>/chunk_0000.wav
skyear dev benchmark --dataset svanstrom --window-sec 1.0 --hf --output-dir reports/benchmark_run
skyear dev simulate
skyear dev simulate fiber-grid --targets 2 --assert-tracks
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
skyear dashboard
skyear-simulate-fiber-grid \
  --server http://127.0.0.1:8080/events \
  --targets 2 \
  --station-failure A4,B2 \
  --assert-tracks
```

The simulator posts synthetic `AcousticEvent` and `StationHeartbeat` JSON only. It does not require real audio. The default layout has three passive lines: A at 800m, B at 600m, and C at 400m from the control point.

Run normal checks:

```bash
python -m compileall station server dashboard shared tools tests
pytest -q
```
