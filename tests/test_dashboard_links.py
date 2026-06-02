from __future__ import annotations

import pytest

from dashboard.station_view import (
    decision_display_state,
    decision_score_values,
    external_spectrum_app_url,
    format_age,
    format_latency,
    format_pct,
    health_badge_label,
    is_event_stale_for_fusion,
    spectrum_page_url,
    timing_summary,
)
from station.station_agent import heartbeat_url_from_events_url


def test_spectrum_page_url_includes_station_id_and_server_url():
    url = spectrum_page_url("sim_001", "http://127.0.0.1:8080")

    assert "01_station_spectrum" in url
    assert "station_id=sim_001" in url
    assert "server_url=http%3A%2F%2F127.0.0.1%3A8080" in url


def test_external_spectrum_app_url_includes_station_id_and_server_url():
    url = external_spectrum_app_url("sim_001", "http://127.0.0.1:8080")

    assert url.startswith("http://localhost:8502/")
    assert "station_id=sim_001" in url
    assert "server_url=http%3A%2F%2F127.0.0.1%3A8080" in url


def test_score_percent_formatting():
    assert format_pct(0.936) == "94%"
    assert format_pct(0.12) == "12%"
    assert format_pct(None) == "n/a"


def test_timing_format_helpers():
    assert format_age(0.25) == "250ms"
    assert format_age(1.2) == "1.2s"
    assert format_latency(0.045) == "45ms"
    assert format_latency(1.25) == "1.25s"


def test_timing_summary_uses_server_received_and_latency():
    summary = timing_summary(
        {
            "timestamp_unix": 100.0,
            "server_received_unix": 101.0,
            "metadata": {"station_to_server_latency_sec": 1.0},
        },
        now=103.0,
    )

    assert summary["event_age"] == "2.0s"
    assert summary["latency"] == "1.00s"


def test_health_badge_shows_no_heartbeat_when_missing():
    assert health_badge_label(None) == "NO HEARTBEAT"
    assert health_badge_label({"alive_state": "offline", "heartbeat": None}) == "NO HEARTBEAT"
    assert health_badge_label({"alive_state": "online", "heartbeat": {"station_id": "s1"}}) == "ONLINE"


def test_heartbeat_url_derived_from_events_url():
    assert heartbeat_url_from_events_url("http://host:8080/events") == "http://host:8080/stations/heartbeat"


def test_decision_display_labels_non_drone_harmonic():
    state = decision_display_state({"metadata": {"harmonic_evidence_pct": 0.95, "ml_drone_pct": 0.01}})

    assert state["label"] == "NON-DRONE HARMONIC"


def test_decision_display_uses_operator_label_for_ml_candidate():
    state = decision_display_state(
        {
            "operator_label": "ml_drone_candidate",
            "harmonic_evidence_pct": 0.23,
            "ml_drone_pct": 0.99,
        }
    )

    assert state["label"] == "ML DRONE CANDIDATE"


def test_decision_scores_include_combined_evidence():
    scores = decision_score_values(
        {
            "harmonic_evidence_pct_smoothed": 0.45,
            "ml_drone_pct": 1.0,
        }
    )

    assert scores["combined"] == pytest.approx(0.62, abs=0.01)
    state = decision_display_state(
        {
            "harmonic_evidence_pct_smoothed": 0.45,
            "ml_drone_pct": 1.0,
            "combined_drone_evidence_pct": scores["combined"],
        }
    )
    assert state["label"] == "STRONG ML DRONE CANDIDATE"


def test_stale_fusion_helper_uses_server_received_time():
    assert is_event_stale_for_fusion({"server_received_unix": 100.0}, fusion_window_sec=8.0, now=109.0)
    assert not is_event_stale_for_fusion({"server_received_unix": 100.0}, fusion_window_sec=8.0, now=107.0)
