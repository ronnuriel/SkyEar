# Datasets And Benchmarks

Dataset and benchmark tools are developer-facing. They are kept out of the quick-start path.

Build a manifest:

```bash
skyear-build-manifest --root data/datasets/<dataset> --output data/manifests/dataset.csv
```

Stream/evaluate a manifest:

```bash
skyear-stream-manifest --manifest data/manifests/dataset.csv --config configs/config_station.yaml --mode offline
```

Run a benchmark:

```bash
skyear dev benchmark --dataset svanstrom --window-sec 1.0 --hf --output-dir reports/benchmark_run
```

Build a recording manifest:

```bash
skyear-build-recording-manifest --root runtime/recordings --output data/manifests/local_recordings_manifest.csv
```

Datasets, generated manifests, reports, and recordings should not be committed unless they are tiny intentional fixtures.
