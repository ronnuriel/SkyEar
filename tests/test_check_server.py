from __future__ import annotations

from tools.check_server import check_server, server_base_url


class FakeResponse:
    def __init__(self, status_code: int = 200, body: dict | None = None, text: str = "ok"):
        self.status_code = status_code
        self._body = body or {"ok": True}
        self.text = text

    def json(self):
        return self._body


def test_server_base_url_accepts_base_or_events_url():
    assert server_base_url("http://server:8080") == "http://server:8080"
    assert server_base_url("http://server:8080/events") == "http://server:8080"
    assert server_base_url("server:8080") == "http://server:8080"


def test_check_server_gets_health_and_skips_heartbeat_without_auth(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(("GET", url))
        return FakeResponse()

    monkeypatch.setattr("tools.check_server.requests.get", fake_get)

    results = check_server("http://server:8080")

    assert all(result.ok for result in results)
    assert calls == [("GET", "http://server:8080/health"), ("GET", "http://server:8080/stations/health")]
    assert results[-1].reason.startswith("skipped")


def test_check_server_posts_heartbeat_when_auth_configured(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(("GET", url))
        return FakeResponse()

    def fake_post(url, json=None, data=None, headers=None, timeout=None):
        calls.append(("POST", url, bool(headers and headers.get("Authorization")), bool(headers and headers.get("X-SkyEar-Signature"))))
        return FakeResponse()

    monkeypatch.setattr("tools.check_server.requests.get", fake_get)
    monkeypatch.setattr("tools.check_server.requests.post", fake_post)

    results = check_server("http://server:8080", api_token="token", hmac_secret="secret")

    assert all(result.ok for result in results)
    assert calls[-1] == ("POST", "http://server:8080/stations/heartbeat", True, True)


def test_check_server_reports_failed_health(monkeypatch):
    def fake_get(url, timeout):
        return FakeResponse(status_code=503, text="down")

    monkeypatch.setattr("tools.check_server.requests.get", fake_get)

    results = check_server("http://server:8080")

    assert results[0].ok is False
    assert "HTTP 503" in results[0].reason
