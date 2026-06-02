from __future__ import annotations

from dashboard.station_view import external_spectrum_app_url, spectrum_page_url


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
