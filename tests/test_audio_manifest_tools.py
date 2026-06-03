from __future__ import annotations

from pathlib import Path

from tools.build_audio_manifest import infer_distance_category, infer_label, infer_source_dataset, iter_manifest_rows
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
    assert summary["false_positives_per_source"] == {"Svanstrom": 1}
