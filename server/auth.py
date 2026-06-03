from __future__ import annotations

import os

from fastapi import Header, HTTPException, Request

from shared.auth import verify_hmac_signature, verify_token


def _token() -> str | None:
    return os.environ.get("SKYEAR_API_TOKEN") or None


def _hmac_secret() -> str | None:
    return os.environ.get("SKYEAR_HMAC_SECRET") or None


async def require_station_auth(
    request: Request,
    authorization: str | None = Header(default=None),
    x_skyear_token: str | None = Header(default=None),
    x_skyear_signature: str | None = Header(default=None),
) -> None:
    token = _token()
    secret = _hmac_secret()
    if not token and not secret:
        return

    body = await request.body()
    token_ok = verify_token(expected_token=token, authorization=authorization, x_token=x_skyear_token)
    hmac_ok = verify_hmac_signature(body, secret, x_skyear_signature)
    if not (token_ok and hmac_ok):
        raise HTTPException(status_code=401, detail="unauthenticated station request")
