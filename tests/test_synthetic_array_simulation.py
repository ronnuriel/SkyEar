from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from server.api import get_map_state, ingest_event
from server.database import db
from server.geo import destination_point, estimate_from_recent_bearings
from station.array_profiles import array_profile
from station.beamforming import bearing_quality_from_result, estimate_bearing
from tools.eval_array_audio import evaluate_array_audio, summarize
from tools.simulate_array_audio import (
    moving_bearings,
    synthesize_array_audio,
    truth_metadata_path,
    write_truth_csv,
    write_truth_metadata,
    write_wav,
)
from tools.simulate_moving_geo import build_moving_geo_event, main as moving_geo_main


def setup_function():
    db.events.clear()
    db.alerts.clear()
    db.heartbeats.clear()


def _angle_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _write_array_config(path: Path) -> None:
    path.write_text(
        """
station: {station_id: sim_array}
audio: {sample_rate: 48000, channels: 8}
detector: {f0_min: 500, max_freq: 7000}
direction:
  enabled: false
  scan_step_deg: 5
  min_beam_confidence_pct: 0.55
  min_peak_ratio: 1.3
  max_second_peak_ratio: 0.85
  reject_ambiguous_bearing: true
mic_array:
  profile: field_8ch_r0_35m
  sync_mode: synchronized
beamforming:
  enabled: true
  method: delay_and_sum
  scan_step_deg: 5
  low_hz: 500
  high_hz: 3000
server: {url: http://127.0.0.1:8080/events}
""",
        encoding="utf-8",
    )


def test_synthetic_8ch_plane_wave_60_deg_estimates_near_60():
    profile = array_profile("field_8ch_r0_35m")
    assert profile is not None
    positions = np.asarray(profile["mic_positions_m"], dtype=np.float64)
    audio, _truth, _metadata = synthesize_array_audio(
        mic_positions_m=positions,
        sample_rate=48000,
        duration_sec=1.0,
        bearing_start_deg=60.0,
        bearing_end_deg=60.0,
        source_type="harmonic_drone",
        f0_hz=1200.0,
        snr_db=25.0,
        noise_type="white",
    )

    result = estimate_bearing(audio, 48000, positions, scan_step_deg=5, low_hz=500, high_hz=3000)

    assert result.bearing_deg is not None
    assert _angle_delta(result.bearing_deg, 60.0) <= 10.0
    assert result.bearing_reliable is True
    assert result.bearing_reject_reason is None
    assert bearing_quality_from_result(result) == "good"


def test_harmonic_interferer_marks_bearing_unreliable():
    profile = array_profile("field_8ch_r0_35m")
    assert profile is not None
    positions = np.asarray(profile["mic_positions_m"], dtype=np.float64)
    audio, _truth, _metadata = synthesize_array_audio(
        mic_positions_m=positions,
        sample_rate=48000,
        duration_sec=1.0,
        bearing_start_deg=60.0,
        bearing_end_deg=60.0,
        source_type="harmonic_drone",
        f0_hz=1200.0,
        snr_db=10.0,
        interferer_type="harmonic",
        interferer_bearing_deg=180.0,
        interferer_f0_hz=1000.0,
        interferer_snr_db=0.0,
        seed=11,
    )

    result = estimate_bearing(audio, 48000, positions, scan_step_deg=5, low_hz=500, high_hz=3000)

    assert result.bearing_deg is not None
    assert _angle_delta(result.bearing_deg, 60.0) > 45.0
    assert result.bearing_reliable is False
    assert result.bearing_reject_reason is not None
    assert "ambiguous" in result.bearing_reject_reason
    assert bearing_quality_from_result(result) == "unreliable"


def test_multipath_degrades_bearing_reliability():
    profile = array_profile("field_8ch_r0_35m")
    assert profile is not None
    positions = np.asarray(profile["mic_positions_m"], dtype=np.float64)
    audio, _truth, _metadata = synthesize_array_audio(
        mic_positions_m=positions,
        sample_rate=48000,
        duration_sec=1.0,
        bearing_start_deg=60.0,
        bearing_end_deg=60.0,
        source_type="harmonic_drone",
        f0_hz=1200.0,
        snr_db=10.0,
        reflection_count=4,
        reflection_delay_ms=[5.0, 12.0, 25.0, 38.0],
        reflection_gain_db=[-3.0, -6.0, -9.0, -12.0],
        reflection_bearing_offset_deg=80.0,
        seed=7,
    )

    result = estimate_bearing(audio, 48000, positions, scan_step_deg=5, low_hz=500, high_hz=3000)

    assert result.bearing_deg is not None
    assert result.bearing_reliable is False
    assert result.bearing_reject_reason is not None
    assert bearing_quality_from_result(result) == "unreliable"


def test_moving_300_to_60_source_produces_monotonic_truth_trend():
    bearings = moving_bearings(start_deg=300.0, end_deg=60.0, sample_count=8)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(bearings)))

    assert list(np.diff(unwrapped)) == pytest.approx([15.0] * 7, abs=1e-6)


def test_eval_array_audio_tracks_synthetic_moving_source(tmp_path: Path):
    profile = array_profile("field_8ch_r0_35m")
    assert profile is not None
    positions = np.asarray(profile["mic_positions_m"], dtype=np.float64)
    wav_path = tmp_path / "moving.wav"
    truth_path = tmp_path / "truth.csv"
    config_path = tmp_path / "config.yaml"
    audio, bearings, _metadata = synthesize_array_audio(
        mic_positions_m=positions,
        sample_rate=48000,
        duration_sec=4.0,
        bearing_start_deg=300.0,
        bearing_end_deg=60.0,
        source_type="harmonic_drone",
        f0_hz=1200.0,
        snr_db=25.0,
        noise_type="white",
    )
    write_wav(wav_path, audio, 48000)
    write_truth_csv(
        truth_path,
        bearings=bearings,
        sample_rate=48000,
        duration_sec=4.0,
        source_type="harmonic_drone",
        snr_db=25.0,
    )
    _write_array_config(config_path)

    rows = evaluate_array_audio(
        wav_path=wav_path,
        truth_path=truth_path,
        config_path=config_path,
        window_sec=1.0,
    )
    metrics = summarize(rows)
    predicted = [row["predicted_bearing_deg"] for row in rows]
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(predicted)))

    assert metrics["median_error_deg"] <= 10.0
    assert all(delta >= -5.0 for delta in np.diff(unwrapped))
    assert "median_confidence" in metrics
    assert "reliable_rate" in metrics
    assert rows[0]["bearing_quality"] in {"good", "fair", "poor", "unreliable"}


def test_simulation_hardening_options_save_truth_metadata(tmp_path: Path):
    profile = array_profile("field_8ch_r0_35m")
    assert profile is not None
    positions = np.asarray(profile["mic_positions_m"], dtype=np.float64)
    truth_path = tmp_path / "truth.csv"

    _audio, bearings, metadata = synthesize_array_audio(
        mic_positions_m=positions,
        sample_rate=48000,
        duration_sec=1.0,
        bearing_start_deg=60.0,
        bearing_end_deg=60.0,
        source_type="harmonic_drone",
        f0_hz=1200.0,
        snr_db=0.0,
        mic_gain_jitter_db=3.0,
        mic_position_jitter_cm=5.0,
        channel_delay_jitter_us=100.0,
        drop_channels=[3],
        permute_channels="random",
        reflection_count=1,
        reflection_delay_ms=[12.0],
        reflection_gain_db=[-8.0],
        interferer_type="harmonic",
        interferer_bearing_deg=180.0,
        wind_noise_level=0.2,
        seed=7,
    )
    write_truth_csv(
        truth_path,
        bearings=bearings,
        sample_rate=48000,
        duration_sec=1.0,
        source_type="harmonic_drone",
        snr_db=0.0,
    )
    metadata_path = write_truth_metadata(truth_path, metadata)

    assert metadata_path == truth_metadata_path(truth_path)
    assert metadata["dropped_channels_before_permutation"] == [3]
    assert sorted(metadata["channel_permutation_output_to_original"]) == list(range(8))
    assert metadata["actual_mic_positions_m"] != metadata["nominal_mic_positions_m"]


def test_eval_array_audio_can_use_wrong_radius_for_config_mismatch(tmp_path: Path):
    profile = array_profile("field_8ch_r0_35m")
    assert profile is not None
    positions = np.asarray(profile["mic_positions_m"], dtype=np.float64)
    wav_path = tmp_path / "moving.wav"
    truth_path = tmp_path / "truth.csv"
    config_path = tmp_path / "config.yaml"
    audio, bearings, _metadata = synthesize_array_audio(
        mic_positions_m=positions,
        sample_rate=48000,
        duration_sec=2.0,
        bearing_start_deg=60.0,
        bearing_end_deg=60.0,
        source_type="harmonic_drone",
        f0_hz=1200.0,
        snr_db=20.0,
        noise_type="white",
    )
    write_wav(wav_path, audio, 48000)
    write_truth_csv(
        truth_path,
        bearings=bearings,
        sample_rate=48000,
        duration_sec=2.0,
        source_type="harmonic_drone",
        snr_db=20.0,
    )
    _write_array_config(config_path)

    rows = evaluate_array_audio(
        wav_path=wav_path,
        truth_path=truth_path,
        config_path=config_path,
        window_sec=1.0,
        array_radius_m=0.12,
    )

    assert rows[0]["beamforming_attempted"] is True
    assert rows[0]["eval_position_source"] == "radius:0.120m"


def test_position_jitter_eval_warns_geometry_mismatch(tmp_path: Path):
    profile = array_profile("field_8ch_r0_35m")
    assert profile is not None
    positions = np.asarray(profile["mic_positions_m"], dtype=np.float64)
    wav_path = tmp_path / "jitter.wav"
    truth_path = tmp_path / "truth.csv"
    config_path = tmp_path / "config.yaml"
    audio, bearings, metadata = synthesize_array_audio(
        mic_positions_m=positions,
        sample_rate=48000,
        duration_sec=1.0,
        bearing_start_deg=60.0,
        bearing_end_deg=60.0,
        source_type="harmonic_drone",
        f0_hz=1200.0,
        snr_db=10.0,
        mic_position_jitter_cm=5.0,
        seed=9,
    )
    write_wav(wav_path, audio, 48000)
    write_truth_csv(
        truth_path,
        bearings=bearings,
        sample_rate=48000,
        duration_sec=1.0,
        source_type="harmonic_drone",
        snr_db=10.0,
    )
    write_truth_metadata(truth_path, metadata)
    _write_array_config(config_path)

    rows = evaluate_array_audio(
        wav_path=wav_path,
        truth_path=truth_path,
        config_path=config_path,
        window_sec=1.0,
    )

    assert rows
    assert str(rows[0]["geometry_warning"]).startswith("simulated_mic_position_mismatch_max_cm=")


def test_synthetic_array_hard_test_writes_summaries(tmp_path: Path):
    subprocess.run(
        ["bash", "scripts/synthetic_array_hard_test.sh", "--output-dir", str(tmp_path)],
        check=True,
        timeout=90,
    )

    txt_path = tmp_path / "hard_test_summary.txt"
    csv_path = tmp_path / "hard_test_summary.csv"
    json_path = tmp_path / "hard_test_summary.json"

    assert txt_path.exists()
    assert csv_path.exists()
    assert json_path.exists()
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert {row["case"] for row in rows} >= {"clean", "harmonic_interferer", "multipath"}
    assert "reliable_rate" in rows[0]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["cases"]


def test_single_channel_config_does_not_attempt_beamforming(tmp_path: Path):
    wav_path = tmp_path / "mono.wav"
    truth_path = tmp_path / "truth.csv"
    config_path = tmp_path / "mono_config.yaml"
    audio = np.zeros((48000, 1), dtype=np.float32)
    write_wav(wav_path, audio, 48000)
    truth_path.write_text("timestamp_sec,bearing_deg,source_type,snr_db\n0,60,tone,20\n1,60,tone,20\n", encoding="utf-8")
    config_path.write_text(
        """
station: {station_id: mono}
audio: {sample_rate: 48000, channels: 1}
detector: {f0_min: 500, max_freq: 7000}
direction: {enabled: false}
mic_array: {profile: mac_builtin_mono}
beamforming: {enabled: true}
server: {url: http://127.0.0.1:8080/events}
""",
        encoding="utf-8",
    )

    rows = evaluate_array_audio(
        wav_path=wav_path,
        truth_path=truth_path,
        config_path=config_path,
        window_sec=1.0,
    )

    assert rows
    assert rows[0]["beamforming_attempted"] is False
    assert rows[0]["predicted_bearing_deg"] is None


def test_two_station_moving_geo_produces_bearing_intersection_estimate():
    now = 1000.0
    event_a = build_moving_geo_event(
        station_id="a",
        station_lat=32.0000,
        station_lon=34.0000,
        target_lat=32.0040,
        target_lon=34.0040,
        step_idx=0,
        timestamp=now,
    )
    event_b = build_moving_geo_event(
        station_id="b",
        station_lat=32.0000,
        station_lon=34.0080,
        target_lat=32.0040,
        target_lon=34.0040,
        step_idx=0,
        timestamp=now,
    )
    ingest_event(event_a)
    ingest_event(event_b)

    state = get_map_state()

    assert len(state["bearing_cues"]) >= 2
    assert state["geo_estimates"]
    assert state["geo_estimates"][0]["estimate_type"] in {"bearing_intersection", "multi_station_area"}


def test_unreliable_bearings_are_not_used_for_geo_fusion():
    now = 1000.0
    events = [
        build_moving_geo_event(
            station_id="a",
            station_lat=32.0000,
            station_lon=34.0000,
            target_lat=32.0040,
            target_lon=34.0040,
            step_idx=0,
            timestamp=now,
        ),
        build_moving_geo_event(
            station_id="b",
            station_lat=32.0000,
            station_lon=34.0080,
            target_lat=32.0040,
            target_lon=34.0040,
            step_idx=0,
            timestamp=now,
        ),
    ]
    for event in events:
        event.bearing_reliable = False
        event.bearing_reject_reason = "high_confidence_ambiguous_lobe"
        event.bearing_quality = "unreliable"
        event.metadata["bearing_reliable"] = False
        ingest_event(event)

    state = get_map_state()

    assert state["bearing_cues"] == []
    assert state["geo_estimates"] == []
    assert {station["bearing_reject_reason"] for station in state["stations"]} == {"high_confidence_ambiguous_lobe"}


def test_bad_geometry_has_lower_geo_confidence():
    now = 1000.0
    target = {"latitude": 32.0, "longitude": 34.0}
    good_a = destination_point(target["latitude"], target["longitude"], 180.0, 500.0)
    good_b = destination_point(target["latitude"], target["longitude"], 90.0, 500.0)
    poor_a = destination_point(target["latitude"], target["longitude"], 180.0, 500.0)
    poor_b = destination_point(target["latitude"], target["longitude"], 190.0, 500.0)

    good = estimate_from_recent_bearings(
        [
            build_moving_geo_event(
                station_id="good_a",
                station_lat=good_a["latitude"],
                station_lon=good_a["longitude"],
                target_lat=target["latitude"],
                target_lon=target["longitude"],
                step_idx=0,
                timestamp=now,
            ),
            build_moving_geo_event(
                station_id="good_b",
                station_lat=good_b["latitude"],
                station_lon=good_b["longitude"],
                target_lat=target["latitude"],
                target_lon=target["longitude"],
                step_idx=0,
                timestamp=now,
            ),
        ],
        now=now + 1.0,
    )
    poor = estimate_from_recent_bearings(
        [
            build_moving_geo_event(
                station_id="poor_a",
                station_lat=poor_a["latitude"],
                station_lon=poor_a["longitude"],
                target_lat=target["latitude"],
                target_lon=target["longitude"],
                step_idx=0,
                timestamp=now,
            ),
            build_moving_geo_event(
                station_id="poor_b",
                station_lat=poor_b["latitude"],
                station_lon=poor_b["longitude"],
                target_lat=target["latitude"],
                target_lon=target["longitude"],
                step_idx=0,
                timestamp=now,
            ),
        ],
        now=now + 1.0,
    )

    assert good["bearing_geometry_quality"] == "good"
    assert poor["bearing_geometry_quality"] == "poor"
    assert poor["confidence"] < good["confidence"]


def test_simulate_moving_geo_cli_posts_two_station_events(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

    monkeypatch.setattr("requests.post", lambda url, json, timeout: calls.append((url, json, timeout)) or Response())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulate_moving_geo",
            "--server",
            "http://server:8080/events",
            "--station-a-lat",
            "32.0",
            "--station-a-lon",
            "34.0",
            "--station-b-lat",
            "32.0",
            "--station-b-lon",
            "34.008",
            "--path-start-lat",
            "32.002",
            "--path-start-lon",
            "34.003",
            "--path-end-lat",
            "32.004",
            "--path-end-lon",
            "34.005",
            "--steps",
            "1",
        ],
    )

    moving_geo_main()

    event_calls = [call for call in calls if call[0].endswith("/events")]
    heartbeat_calls = [call for call in calls if call[0].endswith("/stations/heartbeat")]
    assert len(event_calls) == 2
    assert len(heartbeat_calls) == 2
    assert event_calls[0][1]["estimated_azimuth_deg"] is not None
