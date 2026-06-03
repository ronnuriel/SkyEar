from __future__ import annotations

import csv
import json
import shlex
import sys
from datetime import datetime, timezone

import yaml

from tools.field_session import append_field_note, create_field_session, save_debug_capture
from tools.eval_field_session import evaluate_session
from tools.mark_field_event import main as mark_field_event_main
from tools.mark_field_event import parse_args as parse_mark_field_event_args


def test_start_field_session_creates_expected_files(tmp_path):
    session_dir = create_field_session(
        root=tmp_path,
        session_id="field_001",
        location="north test field",
        station_ids=["station_001", "station_002"],
        weather="clear",
        wind_estimate="light",
        drone_model="DJI_Neo",
        operator_notes="dry run",
        now=datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc),
    )

    assert (session_dir / "session.yaml").exists()
    assert (session_dir / "notes.csv").exists()
    assert (session_dir / "stations").is_dir()
    assert (session_dir / "recordings").is_dir()
    assert (session_dir / "reports").is_dir()

    payload = yaml.safe_load((session_dir / "session.yaml").read_text(encoding="utf-8"))
    assert payload["session_id"] == "field_001"
    assert payload["station_ids"] == ["station_001", "station_002"]
    assert payload["drone_model"] == "DJI_Neo"


def test_mark_field_event_appends_valid_notes_row(tmp_path):
    session_dir = create_field_session(root=tmp_path, session_id="field_002")
    append_field_note(
        session=session_dir,
        label="drone",
        distance_m=50,
        drone_model="DJI_Neo",
        bearing_deg=12.5,
        note="hover 30 sec north",
        timestamp_unix=1000.0,
    )

    with (session_dir / "notes.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["label"] == "drone"
    assert rows[0]["distance_m"] == "50"
    assert rows[0]["drone_model"] == "DJI_Neo"
    assert rows[0]["bearing_deg"] == "12.5"
    assert "timestamp" in rows[0]
    assert "timestamp_unix" not in rows[0]
    assert "ground_truth_bearing_deg" not in rows[0]
    assert rows[0]["note"] == "hover 30 sec north"


def test_mark_field_event_cli_bearing_deg_writes_canonical_field(tmp_path, monkeypatch):
    session_dir = create_field_session(root=tmp_path, session_id="field_cli_bearing")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skyear-mark-field-event",
            "--session",
            str(session_dir),
            "--label",
            "drone",
            "--bearing-deg",
            "42",
        ],
    )

    mark_field_event_main()

    rows = list(csv.DictReader((session_dir / "notes.csv").open("r", encoding="utf-8", newline="")))
    assert rows[0]["bearing_deg"] == "42.0"
    assert "ground_truth_bearing_deg" not in rows[0]


def test_mark_field_event_cli_ground_truth_bearing_alias_writes_canonical_field(tmp_path, monkeypatch):
    session_dir = create_field_session(root=tmp_path, session_id="field_cli_alias")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skyear-mark-field-event",
            "--session",
            str(session_dir),
            "--label",
            "drone",
            "--ground-truth-bearing-deg",
            "90",
        ],
    )

    mark_field_event_main()

    rows = list(csv.DictReader((session_dir / "notes.csv").open("r", encoding="utf-8", newline="")))
    assert rows[0]["bearing_deg"] == "90.0"


def test_eval_field_session_handles_empty_no_drone_session(tmp_path):
    session_dir = create_field_session(root=tmp_path, session_id="field_empty")
    append_field_note(session=session_dir, label="background", note="quiet baseline", timestamp_unix=1000.0)

    summary = evaluate_session(session_dir)

    assert summary["notes"] == 1
    assert summary["windows"] == 0
    assert summary["false_positives_per_hour"] == 0.0
    assert summary["distance_summary"] == {}


def test_eval_field_session_computes_candidate_run2_by_distance(tmp_path):
    session_dir = create_field_session(root=tmp_path, session_id="field_distance")
    append_field_note(session=session_dir, label="drone", distance_m=50, timestamp_unix=1000.0)
    rows = [
        {"timestamp": 1000.0, "operator_label": "background", "ml_drone_pct": 0.1, "window_sec": 1.0},
        {"timestamp": 1001.0, "operator_label": "ml_drone_candidate", "ml_drone_pct": 0.95, "window_sec": 1.0},
        {"timestamp": 1002.0, "operator_label": "ml_drone_candidate", "ml_drone_pct": 0.96, "window_sec": 1.0},
    ]
    history_path = session_dir / "stations" / "station_001_history.jsonl"
    with history_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = evaluate_session(session_dir)
    distance = summary["distance_summary"]["50"]

    assert distance["candidate_any"] is True
    assert distance["candidate_run2"] is True
    assert distance["candidate_run3"] is False
    assert distance["detection_delay_sec"] == 1.0
    assert distance["max_ml"] == 0.96


def test_eval_field_session_reads_old_timestamp_and_ground_truth_bearing(tmp_path):
    session_dir = create_field_session(root=tmp_path, session_id="field_legacy")
    (session_dir / "notes.csv").write_text(
        "timestamp_unix,label,distance_m,ground_truth_bearing_deg,note\n"
        "1000,drone,50,30,legacy note\n",
        encoding="utf-8",
    )
    rows = [
        {
            "timestamp": 1001.0,
            "operator_label": "ml_drone_candidate",
            "estimated_azimuth_deg": 35.0,
            "ml_drone_pct": 0.9,
        },
        {
            "timestamp": 1002.0,
            "operator_label": "ml_drone_candidate",
            "estimated_azimuth_deg": 33.0,
            "ml_drone_pct": 0.95,
        },
    ]
    history_path = session_dir / "stations" / "station_001_history.jsonl"
    with history_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = evaluate_session(session_dir)
    distance = summary["distance_summary"]["50"]

    assert distance["candidate_run2"] is True
    assert distance["detection_delay_sec"] == 1.0
    assert distance["bearing_error_deg_median"] == 4.0


def test_field_alpha_printed_mark_command_is_cli_compatible(monkeypatch):
    script = open("scripts/field_alpha_check.sh", encoding="utf-8").read()
    command = next(line.strip() for line in script.splitlines() if "skyear-mark-field-event" in line and "--bearing-deg" in line)
    command = command.replace("field_sessions/<session_id>", "field_sessions/example")
    monkeypatch.setattr(sys, "argv", shlex.split(command))

    args = parse_mark_field_event_args()

    assert args.bearing_deg == 0.0
    assert args.label == "drone"


def test_save_debug_capture_copies_state_and_recent_history(tmp_path):
    state = tmp_path / "station_latest.json"
    history = tmp_path / "station_history.jsonl"
    state.write_text('{"updated_unix":1000}', encoding="utf-8")
    history.write_text(
        json.dumps({"timestamp": 960.0, "status": "old"}) + "\n"
        + json.dumps({"timestamp": 995.0, "status": "recent"}) + "\n",
        encoding="utf-8",
    )

    capture = save_debug_capture(
        state_path=state,
        history_path=history,
        output_root=tmp_path / "captures",
        seconds=30,
        label="unknown",
        note="manual capture",
        now=1000.0,
    )

    assert (capture / "latest.json").exists()
    assert (capture / "metadata.yaml").exists()
    history_lines = (capture / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1
    assert json.loads(history_lines[0])["status"] == "recent"
