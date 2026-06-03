# SkyEar Dataset Hub

This directory stores dataset metadata, manifests, and evaluation outputs. Raw datasets are intentionally not committed.

## What Is Tracked

- `dataset_registry.yaml`
- `manifests/*.csv`
- `manifests/*.json`
- `manifests/*.jsonl`

## What Is Not Tracked

Raw audio belongs under:

- `data/datasets/`
- `data/raw/`

Large WAV/FLAC/MP3/M4A/OGG files are ignored by Git.

## Basic Workflow

List registered datasets:

```bash
skyear-datasets list
```

Download or prepare a dataset:

```bash
skyear-download-datasets --dataset droneaudioset_hf
```

Build a manifest:

```bash
skyear-build-manifest \
  --registry data/dataset_registry.yaml \
  --dataset droneaudioset_hf \
  --output data/manifests/all_sources_manifest.csv
```

Run the offline station detector:

```bash
skyear-stream-manifest \
  --manifest data/manifests/all_sources_manifest.csv \
  --config configs/config_station.yaml \
  --mode offline \
  --window-sec 1.0 \
  --save-report reports/manifest_eval.csv
```

Summarize benchmark results:

```bash
skyear-summarize-benchmark \
  --report reports/manifest_eval.csv \
  --output reports/benchmark_summary.json
```

## License And Citation Responsibility

Each public dataset may have different licensing, citation, and redistribution requirements. The registry records known status and notes, but the operator is responsible for reviewing the upstream dataset license before training, sharing, or publishing derived results.

## Operational Caveat

Public datasets are useful for engineering benchmarks, false-positive analysis, and training candidates. They do not validate operational field performance by themselves. Field sessions, local calibration, station health, microphone geometry, weather, and ground-truth notes are still required.
