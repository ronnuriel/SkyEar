from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import requests

from shared.auth import auth_headers


@dataclass
class CheckResult:
    name: str
    ok: bool
    reason: str


def server_base_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        parsed = urlparse("http://" + str(url).strip())
    path = parsed.path.rstrip("/")
    if path.endswith("/events"):
        path = path[: -len("/events")]
    return urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "", "")).rstrip("/")


def _get_json(url: str, timeout: float) -> tuple[bool, str]:
    try:
        response = requests.get(url, timeout=timeout)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if response.status_code >= 400:
        return False, f"HTTP {response.status_code}: {response.text[:200]}"
    try:
        response.json()
    except Exception as exc:
        return False, f"invalid JSON: {type(exc).__name__}: {exc}"
    return True, f"HTTP {response.status_code}"


def _post_heartbeat(base_url: str, timeout: float, api_token: str | None, hmac_secret: str | None) -> CheckResult:
    payload = {
        "station_id": "skyear_connectivity_check",
        "station_name": "SkyEar connectivity check",
        "timestamp_unix": time.time(),
        "status": "online",
        "metadata": {"source": "skyear-check-server"},
    }
    headers = auth_headers(payload, api_token=api_token, hmac_secret=hmac_secret)
    try:
        if hmac_secret:
            body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            headers["Content-Type"] = "application/json"
            response = requests.post(f"{base_url}/stations/heartbeat", data=body, headers=headers, timeout=timeout)
        else:
            response = requests.post(f"{base_url}/stations/heartbeat", json=payload, headers=headers, timeout=timeout)
    except Exception as exc:
        return CheckResult("POST /stations/heartbeat", False, f"{type(exc).__name__}: {exc}")
    if response.status_code >= 400:
        return CheckResult("POST /stations/heartbeat", False, f"HTTP {response.status_code}: {response.text[:200]}")
    return CheckResult("POST /stations/heartbeat", True, f"HTTP {response.status_code}")


def check_server(
    url: str,
    *,
    timeout: float = 3.0,
    api_token: str | None = None,
    hmac_secret: str | None = None,
    post_heartbeat: bool | None = None,
) -> list[CheckResult]:
    base_url = server_base_url(url)
    results: list[CheckResult] = []
    for name, path in (("GET /health", "/health"), ("GET /stations/health", "/stations/health")):
        ok, reason = _get_json(f"{base_url}{path}", timeout)
        results.append(CheckResult(name, ok, reason))

    should_post_heartbeat = bool(api_token or hmac_secret) if post_heartbeat is None else bool(post_heartbeat)
    if should_post_heartbeat:
        results.append(_post_heartbeat(base_url, timeout, api_token, hmac_secret))
    else:
        results.append(CheckResult("POST /stations/heartbeat", True, "skipped; no token/HMAC configured"))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check connectivity from a station host to a SkyEar server.")
    parser.add_argument("--url", required=True, help="Server base URL or events URL, for example http://SERVER:8080")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--api-token", default=os.environ.get("SKYEAR_API_TOKEN"))
    parser.add_argument("--hmac-secret", default=os.environ.get("SKYEAR_HMAC_SECRET"))
    parser.add_argument("--post-heartbeat", action="store_true", help="POST a test heartbeat even without auth configured.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = check_server(
        args.url,
        timeout=args.timeout,
        api_token=args.api_token,
        hmac_secret=args.hmac_secret,
        post_heartbeat=True if args.post_heartbeat else None,
    )
    failed = False
    for result in results:
        prefix = "OK" if result.ok else "FAILED"
        print(f"{prefix}: {result.name} - {result.reason}")
        failed = failed or not result.ok
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
