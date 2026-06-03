from __future__ import annotations

from pathlib import Path

from tools.build_audio_manifest import infer_distance_category, infer_label, infer_label_details, infer_source_dataset, iter_manifest_rows
from tools.eval_manifest_dataset import file_level_metrics, merge_predictions_with_manifest, summarize


def test_manifest_infers_dataset_label_distance(tmp_path: Path):
    audio = tmp_path / "svanstrom" / "Distant" / "drone" / "clip.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"")

    rows = list(iter_manifest_rows([tmp_path]))

    assert len(rows) == 1
    assert rows[0]["source_dataset"] == "svanstrom"
    assert rows[0]["label"] == "drone"
    assert rows[0]["distance_category"] == "distant"
    assert infer_label(audio) == "drone"
    assert infer_distance_category(audio) == "distant"
    assert infer_source_dataset(audio) == "svanstrom"


def test_manifest_negative_labels_win_before_drone():
    details = infer_label_details(Path("not_drone/foo.wav"))

    assert details["label"] == "background"
    assert details["label_source"] == "negative_path_token"


def test_manifest_droneaudioset_background_is_not_drone():
    details = infer_label_details(Path("DroneAudioSet/background/foo.wav"))

    assert details["label"] == "background"


def test_manifest_svanstrom_drone_filename_is_drone():
    details = infer_label_details(Path("svanstrom/DRONE_001.wav"))

    assert details["label"] == "drone"
    assert details["label_source"] == "svanstrom_filename"


def test_manifest_svanstrom_helicopter_filename_is_helicopter():
    details = infer_label_details(Path("svanstrom/HELICOPTER_001.wav"))

    assert details["label"] == "helicopter"


def test_eval_manifest_computes_run2_run3_and_false_positive_source():
    manifest = [
        {"file_path": "drone.wav", "label": "drone", "source_dataset": "Svanstrom", "split_group": "holdout"},
        {"file_path": "background.wav", "label": "background", "source_dataset": "Svanstrom", "split_group": "holdout"},
    ]
    predictions = [
        {"file_path": "drone.wav", "window_idx": 0, "operator_label": "ml_drone_candidate", "status": "suspect"},
        {"file_path": "drone.wav", "window_idx": 1, "operator_label": "local_drone_candidate", "status": "suspect"},
        {"file_path": "drone.wav", "window_idx": 2, "operator_label": "strong_local_candidate", "status": "suspect"},
        {"file_path": "background.wav", "window_idx": 0, "operator_label": "background", "status": "background"},
        {"file_path": "background.wav", "window_idx": 1, "operator_label": "ml_drone_candidate", "status": "suspect"},
    ]

    rows = merge_predictions_with_manifest(manifest, predictions)
    files = file_level_metrics(rows)
    summary = summarize(rows)

    drone_file = next(row for row in files if row["file_path"] == "drone.wav")
    assert drone_file["candidate_any"] is True
    assert drone_file["candidate_run2"] is True
    assert drone_file["candidate_run3"] is True
    assert drone_file["strong_any"] is True
    assert drone_file["max_candidate_run"] == 3
    assert summary["false_positives_per_source"] == {"Svanstrom": 1}
    assert "false_candidate_run2_rate" in summary["file_level"]
    assert "source_dataset" in summary["grouped"]


def test_eval_manifest_computes_time_and_max_scores():
    rows = merge_predictions_with_manifest(
        [{"file_path": "drone.wav", "label": "drone", "source_dataset": "DroneAudioSet"}],
        [
            {
                "file_path": "drone.wav",
                "window_idx": 0,
                "window_sec": 1.0,
                "operator_label": "background",
                "status": "background",
                "ml_drone_pct": 0.2,
                "harmonic_evidence_pct_smoothed": 0.1,
                "combined_drone_evidence_pct": 0.1,
            },
            {
                "file_path": "drone.wav",
                "window_idx": 1,
                "window_sec": 1.0,
                "operator_label": "ml_drone_candidate",
                "status": "suspect",
                "ml_drone_pct": 0.9,
                "harmonic_evidence_pct_smoothed": 0.5,
                "combined_drone_evidence_pct": 0.64,
            },
        ],
    )

    file_row = file_level_metrics(rows)[0]

    assert file_row["time_to_first_candidate_sec"] == 1.0
    assert file_row["max_ml"] == 0.9
    assert file_row["max_harmonic"] == 0.5
    assert file_row["max_combined"] == 0.64
