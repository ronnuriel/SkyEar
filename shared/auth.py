from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def auth_headers(payload: Any, api_token: str | None = None, hmac_secret: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    if hmac_secret:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers["X-SkyEar-Signature"] = "sha256=" + hmac.new(
            hmac_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
    return headers


def verify_token(
    *,
    expected_token: str | None,
    authorization: str | None,
    x_token: str | None,
) -> bool:
    if not expected_token:
        return True
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    return hmac.compare_digest(expected_token, bearer) or hmac.compare_digest(expected_token, x_token or "")


def verify_hmac_signature(body: bytes, expected_secret: str | None, signature: str | None) -> bool:
    if not expected_secret:
        return True
    if not signature:
        return False
    expected = "sha256=" + hmac.new(expected_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
