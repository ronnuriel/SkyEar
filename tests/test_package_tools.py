from __future__ import annotations

import sys

import pytest

from station import station_agent
from tools.copy_configs import copy_configs
from tools.copy_configs import main as copy_configs_main


def test_copy_configs_copies_packaged_yaml_files(tmp_path):
    copied = copy_configs(tmp_path)

    names = {path.name for path in copied}
    assert "config_station.yaml" in names
    assert "config_station_2.yaml" in names
    assert "config_station_array_8ch.yaml" in names
    assert "config_station_remote.yaml" in names
    assert (tmp_path / "config_station.yaml").read_text(encoding="utf-8").startswith("station:")


def test_copy_configs_cli_accepts_positional_destination(tmp_path, monkeypatch):
    destination = tmp_path / "positional_configs"
    monkeypatch.setattr(sys, "argv", ["skyear-copy-configs", str(destination)])

    copy_configs_main()

    assert (destination / "config_station.yaml").exists()
    assert (destination / "config_station_remote.yaml").exists()


def test_copy_configs_cli_accepts_output_destination(tmp_path, monkeypatch):
    destination = tmp_path / "output_configs"
    monkeypatch.setattr(sys, "argv", ["skyear-copy-configs", "--output", str(destination)])

    copy_configs_main()

    assert (destination / "config_station.yaml").exists()
    assert (destination / "config_station_remote.yaml").exists()


def test_station_default_missing_config_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["skyear-station"])

    with pytest.raises(SystemExit) as exc:
        station_agent.main()

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "Default config not found. Run: skyear-copy-configs ./configs or pass --config PATH" in captured.err
    assert "Traceback" not in captured.err
