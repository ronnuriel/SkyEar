from __future__ import annotations

import yaml


def test_array_8ch_config_radius_matches_explicit_positions():
    cfg = yaml.safe_load(open("configs/config_station_array_8ch.yaml", encoding="utf-8"))

    positions = cfg["mic_array"]["mic_positions_m"]
    assert len(positions) == 8
    assert cfg["mic_array"]["profile"] == "field_8ch_r0_35m"
    assert cfg["direction"]["array_radius_m"] == 0.35
    assert max(abs(float(x)) for x, _y, _z in positions) == 0.35
    assert cfg["beamforming"]["low_hz"] == 500
    assert cfg["beamforming"]["high_hz"] == 3000
    assert cfg["detector"]["max_freq"] == 7000
