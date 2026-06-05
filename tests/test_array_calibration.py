from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from station import station_agent
from station.array_calibration import (
    apply_array_calibration,
    build_calibration,
    detect_strongest_channel,
    estimate_delay_correction_samples,
    estimate_gain_correction,
    load_calibration,
    validate_calibration_payload,
    verify_channel_order,
)
from station.direction import fractional_delay


def test_wrong_channel_order_detected():
    detected = [0, 2, 1, 3]

    report = verify_channel_order(detected, channel_count=4)

    assert report["correct"] is False
    assert report["suggested_input_channel_order"] == detected


def test_detect_strongest_channel_for_tap():
    audio = np.zeros((1000, 4), dtype=np.float32)
    audio[200, 2] = 1.0

    assert detect_strongest_channel(audio) == 2


def test_gain_mismatch_corrected():
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 0.1, size=4800).astype(np.float32)
    audio = np.stack([base, base * 0.5, base * 2.0], axis=1)
    gains, _rms, _health = estimate_gain_correction(audio)
    corrected, _positions, _metadata = apply_array_calibration(
        audio,
        None,
        {
            "channel_count": 3,
            "input_channel_order": [0, 1, 2],
            "gain_correction": gains,
            "delay_correction_samples": [0.0, 0.0, 0.0],
            "bad_channels": [],
            "channel_rms": [0.1, 0.05, 0.2],
            "channel_health": ["ok", "ok", "ok"],
            "calibration_valid": True,
        },
    )

    rms = np.sqrt(np.mean(corrected.astype(np.float64) ** 2, axis=0))
    assert float(np.max(rms) - np.min(rms)) < 1e-5


def test_delay_mismatch_corrected():
    signal = np.zeros(2000, dtype=np.float32)
    signal[500] = 1.0
    signal[700] = 0.5
    delayed = fractional_delay(signal, 5.0).astype(np.float32)
    audio = np.stack([signal, delayed], axis=1)

    corrections = estimate_delay_correction_samples(audio, max_lag_samples=20)
    corrected, _positions, _metadata = apply_array_calibration(
        audio,
        None,
        {
            "channel_count": 2,
            "input_channel_order": [0, 1],
            "gain_correction": [1.0, 1.0],
            "delay_correction_samples": corrections,
            "bad_channels": [],
            "channel_rms": [1.0, 1.0],
            "channel_health": ["ok", "ok"],
            "calibration_valid": True,
        },
    )

    assert corrections == [0.0, -5.0]
    assert int(np.argmax(corrected[:, 0])) == int(np.argmax(corrected[:, 1]))
    assert float(np.max(np.abs(corrected[:, 0] - corrected[:, 1]))) < 1e-6


def test_dropped_channel_excluded():
    audio = np.ones((100, 4), dtype=np.float32)
    positions = np.asarray([[idx, 0.0, 0.0] for idx in range(4)], dtype=np.float64)

    corrected, corrected_positions, metadata = apply_array_calibration(
        audio,
        positions,
        {
            "channel_count": 4,
            "input_channel_order": [0, 1, 2, 3],
            "gain_correction": [1.0, 1.0, 1.0, 1.0],
            "delay_correction_samples": [0.0, 0.0, 0.0, 0.0],
            "bad_channels": [2],
            "channel_rms": [1.0, 1.0, 1.0, 1.0],
            "channel_health": ["ok", "ok", "bad", "ok"],
            "calibration_valid": True,
        },
    )

    assert corrected.shape[1] == 3
    assert corrected_positions is not None
    assert corrected_positions.shape[0] == 3
    assert metadata["kept_channels"] == [0, 1, 3]


def test_build_calibration_marks_dropout_channel():
    rng = np.random.default_rng(9)
    audio = rng.normal(0.0, 0.1, size=(4800, 4)).astype(np.float32)
    audio[:, 3] = 0.0

    calibration = build_calibration(audio, sample_rate=48000, estimate_delay=False)

    assert 3 in calibration["bad_channels"]
    assert calibration["channel_health"][3] == "silent"
    assert calibration["calibration_valid"] is False


def test_zero_rms_calibration_is_invalid():
    payload = {
        "channel_count": 8,
        "channel_rms": [0.0] * 8,
        "channel_health": ["ok"] * 8,
        "bad_channels": [],
    }

    validation = validate_calibration_payload(payload)

    assert validation["calibration_valid"] is False
    assert validation["silent_channels"] == list(range(8))


def test_zero_rms_channels_are_not_ok(tmp_path: Path):
    path = tmp_path / "placeholder.json"
    path.write_text(
        json.dumps(
            {
                "channel_count": 2,
                "input_channel_order": [0, 1],
                "gain_correction": [1.0, 1.0],
                "delay_correction_samples": [0.0, 0.0],
                "bad_channels": [],
                "channel_rms": [0.0, 0.0],
                "channel_health": ["ok", "ok"],
            }
        ),
        encoding="utf-8",
    )

    calibration = load_calibration(path)

    assert calibration is not None
    assert calibration.calibration_valid is False
    assert calibration.calibration_type == "placeholder"
    assert calibration.channel_health == ["silent", "silent"]


def test_8ch_without_calibration_warns(capsys):
    calibration = station_agent.load_array_calibration_for_station({}, {"channels": 8})

    output = capsys.readouterr().out
    assert calibration is None
    assert "Array uncalibrated; bearing may be unreliable" in output


def test_load_array_calibration_from_file(tmp_path: Path, capsys):
    path = tmp_path / "cal.json"
    path.write_text(
        json.dumps(
            {
                "channel_count": 2,
                "input_channel_order": [1, 0],
                "gain_correction": [1.0, 1.0],
                "delay_correction_samples": [0.0, 0.0],
                "bad_channels": [],
                "channel_rms": [0.1, 0.1],
                "channel_health": ["ok", "ok"],
                "calibration_valid": True,
            }
        ),
        encoding="utf-8",
    )

    calibration = station_agent.load_array_calibration_for_station(
        {"calibration_file": str(path)},
        {"channels": 8},
    )

    assert calibration is not None
    assert calibration.input_channel_order == [1, 0]
    assert "Array calibration loaded" in capsys.readouterr().out


def test_8ch_placeholder_calibration_warns_invalid(tmp_path: Path, capsys):
    path = tmp_path / "placeholder.json"
    path.write_text(
        json.dumps(
            {
                "calibration_type": "placeholder",
                "calibration_valid": False,
                "channel_count": 8,
                "input_channel_order": list(range(8)),
                "gain_correction": [1.0] * 8,
                "delay_correction_samples": [0.0] * 8,
                "bad_channels": [],
                "channel_rms": [0.0] * 8,
                "channel_health": ["ok"] * 8,
            }
        ),
        encoding="utf-8",
    )

    calibration = station_agent.load_array_calibration_for_station(
        {"calibration_file": str(path)},
        {"channels": 8},
    )

    output = capsys.readouterr().out
    assert calibration is not None
    assert calibration.calibration_valid is False
    assert "array calibration is placeholder/invalid" in output
    assert "Array uncalibrated; bearing may be unreliable" in output
