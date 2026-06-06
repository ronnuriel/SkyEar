from __future__ import annotations

from pathlib import Path

import numpy as np

from station.recording_manager import RecordingManager
from tools.cli import _extract_config, _version_from_tag, main


def test_skyear_help_prints_command_groups(capsys):
    assert main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "skyear station" in output
    assert "skyear rec start <session>" in output
    assert "skyear release preflight" in output


def test_cli_config_override_parsing(monkeypatch):
    monkeypatch.setenv("SKYEAR_CONFIG", "configs/from_env.yaml")

    config, argv = _extract_config(["--config", "configs/explicit.yaml", "station"])

    assert config == "configs/explicit.yaml"
    assert argv == ["station"]


def test_cli_check_two_mic_dry_run_uses_config_default(tmp_path: Path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
audio:
  device_id: 3
  sample_rate: 48000
  channels: 2
two_mic_direction:
  spacing_m: 2.0
  center_deadzone_deg: 12
""".strip(),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "check", "two-mic", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "Two-mic direction check" in output
    assert "device_id=3" in output


def test_cli_recording_summary_uses_config_root(tmp_path: Path, capsys):
    root = tmp_path / "recordings"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"recording:\n  root: {root}\n", encoding="utf-8")
    manager = RecordingManager(
        station_id="station_test",
        sample_rate=8000,
        channels=1,
        config={"enabled": True, "root": str(root), "chunk_sec": 1, "max_disk_gb": 20},
    )
    manager.start_recording("cli_summary")
    manager.append_audio(np.zeros((8000, 1), dtype=np.float32), timestamp=100.0)
    manager.stop_recording()

    assert main(["--config", str(config_path), "rec", "summary", "--json"]) == 0

    output = capsys.readouterr().out
    assert '"session_id": "station_test_' in output
    assert '"total_wav_duration_sec": 1.0' in output


def test_release_tag_version_parser():
    assert _version_from_tag("v0.2.0-field-alpha") == "0.2.0"
    assert _version_from_tag("0.2.1") == "0.2.1"
