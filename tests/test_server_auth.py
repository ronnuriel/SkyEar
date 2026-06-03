from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException

from server.auth import require_station_auth
from shared.auth import auth_headers


class FakeRequest:
    def __init__(self, body: bytes):
        self._body = body

    async def body(self) -> bytes:
        return self._body


def _run_auth(
    request: FakeRequest,
    authorization: str | None = None,
    x_skyear_token: str | None = None,
    x_skyear_signature: str | None = None,
):
    return asyncio.run(
        require_station_auth(
            request,
            authorization=authorization,
            x_skyear_token=x_skyear_token,
            x_skyear_signature=x_skyear_signature,
        )
    )


def test_station_auth_rejects_missing_token_when_configured(monkeypatch):
    monkeypatch.setenv("SKYEAR_API_TOKEN", "station-token")
    monkeypatch.delenv("SKYEAR_HMAC_SECRET", raising=False)

    with pytest.raises(HTTPException) as exc:
        _run_auth(FakeRequest(b"{}"))

    assert exc.value.status_code == 401


def test_station_auth_accepts_station_token(monkeypatch):
    monkeypatch.setenv("SKYEAR_API_TOKEN", "station-token")
    monkeypatch.delenv("SKYEAR_HMAC_SECRET", raising=False)

    assert _run_auth(FakeRequest(b"{}"), authorization="Bearer station-token") is None


def test_station_auth_accepts_hmac_signature(monkeypatch):
    monkeypatch.delenv("SKYEAR_API_TOKEN", raising=False)
    monkeypatch.setenv("SKYEAR_HMAC_SECRET", "station-secret")
    payload = {"station_id": "auth_station", "timestamp_unix": 100.0, "status": "online"}
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    headers = auth_headers(payload, hmac_secret="station-secret")

    assert _run_auth(FakeRequest(body), x_skyear_signature=headers["X-SkyEar-Signature"]) is None
