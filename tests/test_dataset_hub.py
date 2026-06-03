from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

from tools.build_audio_manifest import iter_manifest_rows, parse_droneaudioset_path
from tools.build_training_splits import build_splits, group_key
from tools.dataset_registry import DEFAULT_REGISTRY, dataset_by_id, validate_registry
from tools.eval_audio import main as eval_audio_main
from tools.stream_manifest_dataset import run_manifest, run_single_wav
from tools.summarize_benchmark import summarize_report


def _write_tone(path: Path, sample_rate: int = 16000, duration_sec: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
    audio = (0.05 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    wavfile.write(path, sample_rate, audio)


def _write_config(path: Path, sample_rate: int = 16000) -> None:
    payload = {
        "station": {"station_id": "offline_test"},
        "audio": {"channels": 1, "sample_rate": sample_rate, "window_sec": 1.0},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_dataset_registry_validates_seed_file():
    assert validate_registry(DEFAULT_REGISTRY) == []
    dataset = dataset_by_id("droneaudioset_hf", DEFAULT_REGISTRY)
    assert dataset["metadata_parser"] == "droneaudioset"


def test_manifest_parses_droneaudioset_path_metadata():
    path = Path("DroneAudioSet/drone-only-recordings/drone2-only/mic-dist-50cm/throttle-low/mic2_8array-down-File3.wav")
    metadata = parse_droneaudioset_path(path)

    assert metadata["drone_id"] == "drone2"
    assert metadata["distance_m"] == 0.5
    assert metadata["throttle"] == "low"
    assert metadata["mic_id"] == "mic2"
    assert metadata["array_info"] == "8array-down"


def test_manifest_rows_include_new_columns_and_svanstrom_labels(tmp_path):
    audio = tmp_path / "svanstrom" / "DRONE_001.wav"
    _write_tone(audio)

    rows = list(iter_manifest_rows([tmp_path], verify_audio=True, skip_audio_hash=True))

    assert rows[0]["audio_path"].endswith("DRONE_001.wav")
    assert rows[0]["label"] == "drone"
    assert rows[0]["label_source"] == "svanstrom_filename"
    assert rows[0]["sample_rate"] == 16000
    assert rows[0]["duration_sec"] == 2.0


def test_stream_manifest_dataset_runs_on_tiny_wav(tmp_path):
    wav_path = tmp_path / "dataset" / "drone.wav"
    _write_tone(wav_path)
    config = tmp_path / "config.yaml"
    _write_config(config)
    manifest = tmp_path / "manifest.csv"
    report = tmp_path / "report.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "file_path", "dataset_id", "source_dataset", "label"])
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": str(wav_path),
                "file_path": str(wav_path),
                "dataset_id": "synthetic",
                "source_dataset": "synthetic",
                "label": "drone",
            }
        )

    events = run_manifest(
        argparse.Namespace(
            manifest=manifest,
            config=config,
            mode="offline",
            window_sec=1.0,
            save_report=report,
            server=None,
            dataset=[],
            max_files=None,
            max_windows=1,
            hf=False,
            model_id="unused",
        )
    )

    rows = list(csv.DictReader(report.open("r", encoding="utf-8", newline="")))
    assert len(events) == 1
    assert rows[0]["dataset_id"] == "synthetic"
    assert rows[0]["label"] == "drone"
    assert "combined_drone_evidence_pct" in rows[0]


def test_eval_audio_produces_report_rows(tmp_path):
    wav_path = tmp_path / "single.wav"
    _write_tone(wav_path)
    config = tmp_path / "config.yaml"
    _write_config(config)
    report = tmp_path / "single_report.csv"

    events = run_single_wav(wav=wav_path, config=config, label="drone", save_report=report, window_sec=1.0)

    rows = list(csv.DictReader(report.open("r", encoding="utf-8", newline="")))
    assert events
    assert rows[0]["dataset_id"] == "single_wav"
    assert rows[0]["label"] == "drone"


def test_benchmark_summary_counts_false_runs():
    rows = [
        {"file_path": "bg.wav", "dataset_id": "synthetic", "label": "background", "window_idx": "0", "operator_label": "background"},
        {
            "file_path": "bg.wav",
            "dataset_id": "synthetic",
            "label": "background",
            "window_idx": "1",
            "operator_label": "ml_drone_candidate",
        },
        {
            "file_path": "bg.wav",
            "dataset_id": "synthetic",
            "label": "background",
            "window_idx": "2",
            "operator_label": "ml_drone_candidate",
        },
    ]

    summary = summarize_report(rows)

    assert summary["overall"]["candidate_run2"] == 1
    assert summary["overall"]["false_candidate_events_per_hour"] > 0
    assert summary["overall"]["confusion_operator_label"]["ml_drone_candidate"] == 2


def test_split_builder_does_not_split_same_source_file():
    rows = [
        {"dataset_id": "a", "original_dataset_path": "same.wav", "audio_path": "same.wav", "drone_id": "", "environment": ""},
        {"dataset_id": "a", "original_dataset_path": "same.wav", "audio_path": "same.wav", "drone_id": "", "environment": ""},
        {"dataset_id": "a", "original_dataset_path": "other.wav", "audio_path": "other.wav", "drone_id": "", "environment": ""},
    ]

    splits = build_splits(rows)
    locations = []
    for split, split_rows in splits.items():
        if split == "holdout_by_source":
            continue
        if any(group_key(row) == group_key(rows[0]) for row in split_rows):
            locations.append(split)

    assert len(locations) == 1
