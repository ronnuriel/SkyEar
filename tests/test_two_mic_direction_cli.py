from __future__ import annotations

from pathlib import Path

from tools.check_two_mic_direction import main


def test_check_two_mic_direction_dry_run_prints_operator_instructions(tmp_path: Path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
audio:
  device_id: 7
  sample_rate: 48000
  channels: 2
  window_sec: 1.0
two_mic_direction:
  spacing_m: 2.0
  left_channel: 0
  right_channel: 1
  center_deadzone_deg: 12
""".strip(),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "Clap near the left mic -> should show LOOK LEFT" in output
    assert "Clap near the right mic -> should show LOOK RIGHT" in output
    assert "Clap centered in front -> should show LOOK CENTER" in output
    assert "front/back ambiguous" in output
    assert "spacing_m=2" in output
