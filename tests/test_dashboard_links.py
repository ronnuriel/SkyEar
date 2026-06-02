from __future__ import annotations

from dashboard.station_view import decision_display_state, format_pct, external_spectrum_app_url, spectrum_page_url


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
