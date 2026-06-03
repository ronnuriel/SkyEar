from __future__ import annotations

from tools.copy_configs import copy_configs


def test_copy_configs_copies_packaged_yaml_files(tmp_path):
    copied = copy_configs(tmp_path)

    names = {path.name for path in copied}
    assert "config_station.yaml" in names
    assert "config_station_2.yaml" in names
    assert "config_station_array_8ch.yaml" in names
    assert "config_station_remote.yaml" in names
    assert (tmp_path / "config_station.yaml").read_text(encoding="utf-8").startswith("station:")
