from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

from tools.combine_benchmark_summaries import COMBINED_FIELDNAMES, combine_summaries
from tools.release_checks import git_check_ignore, tracked_raw_audio_files
from tools.run_dataset_benchmarks import run_benchmarks


def _write_tone(path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.05 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    wavfile.write(path, sample_rate, audio)


def _write_config(path: Path, sample_rate: int = 16000) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "station": {"station_id": "benchmark_test"},
                "audio": {"channels": 1, "sample_rate": sample_rate, "window_sec": 1.0},
            }
        ),
        encoding="utf-8",
    )


def test_run_dataset_benchmarks_works_on_synthetic_fixture(tmp_path):
    dataset_root = tmp_path / "fixtures" / "svanstrom"
    _write_tone(dataset_root / "Close" / "drone" / "DRONE_001.wav")
    _write_tone(dataset_root / "Distant" / "background" / "BACKGROUND_001.wav")
    config = tmp_path / "config.yaml"
    _write_config(config)
    output_dir = tmp_path / "reports"

    payload = run_benchmarks(
        argparse.Namespace(
            registry=Path("data/dataset_registry.yaml"),
            dataset=["synthetic_fixture"],
            datasets=[],
            root=[f"synthetic_fixture={dataset_root}"],
            config=config,
            output_dir=output_dir,
            window_sec=1.0,
            hf=False,
            model_id="unused",
            max_files=None,
            max_windows=1,
            verify_audio=False,
        )
    )

    result = payload["results"][0]
    summary_path = output_dir / "synthetic_fixture" / "summary.json"
    assert result["ok"] is True
    assert (output_dir / "synthetic_fixture" / "eval.csv").exists()
    assert summary_path.exists()
    assert (output_dir / "all_benchmarks" / "run_summary.json").exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["overall"]["files"] == 2


def test_combined_summary_csv_has_expected_columns(tmp_path):
    dataset_dir = tmp_path / "reports" / "synthetic"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "summary.json").write_text(
        json.dumps(
            {
                "overall": {"files": 2, "false_alert_file_rate": 0.0, "median_time_to_first_candidate": 1.0},
                "by_label": {
                    "drone": {"files": 1, "candidate_run2_files": 1, "candidate_run3_files": 0, "strong_run2_files": 1},
                    "background": {"files": 1, "false_candidate_run2_files": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    rows = combine_summaries(tmp_path / "reports")

    assert rows[0]["dataset_id"] == "synthetic"
    assert set(COMBINED_FIELDNAMES) == set(rows[0])
    assert rows[0]["drone_candidate_run2_rate"] == 1.0


def test_release_check_scripts_pass_syntax():
    for script in ("scripts/release_field_alpha_check.sh", "scripts/run_dataset_benchmarks.sh"):
        subprocess.run(["bash", "-n", script], check=True)


def test_no_raw_audio_tracked_check_works(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    (tmp_path / "clip.wav").write_bytes(b"not really audio")
    (tmp_path / "README.md").write_text("ok\n", encoding="utf-8")
    subprocess.run(["git", "add", "clip.wav", "README.md"], cwd=tmp_path, check=True)

    assert tracked_raw_audio_files(tmp_path) == ["clip.wav"]


def test_git_check_ignore_detects_ignored_path(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    (tmp_path / ".gitignore").write_text("data/datasets/\n", encoding="utf-8")

    assert git_check_ignore("data/datasets/example.wav", tmp_path) is True
