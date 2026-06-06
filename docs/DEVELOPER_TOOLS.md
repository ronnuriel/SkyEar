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
```

Legacy scripts remain available for compatibility, including:

- `skyear-stream-local-dataset`
- `skyear-stream-hf-dataset`
- `skyear-build-manifest`
- `skyear-eval-manifest`
- `skyear-run-benchmarks`
- `skyear-simulate-geo-events`
- `skyear-simulate-array-audio`
- `skyear-debug-harmonic-wav`

Run normal checks:

```bash
python -m compileall station server dashboard shared tools tests
pytest -q
```
