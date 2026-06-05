from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from station.recording_manager import RecordingManager


class RecordingControlServer:
    def __init__(self, manager: RecordingManager, host: str = "127.0.0.1", port: int = 8765):
        self.manager = manager
        self.host = str(host)
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        manager = self.manager

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") == "/recording/state":
                    _write_json(self, 200, manager.state())
                    return
                _write_json(self, 404, {"ok": False, "error": "not_found"})

            def do_POST(self):  # noqa: N802
                payload = _read_json(self)
                path = self.path.rstrip("/")
                if path == "/recording/start":
                    state = manager.start_recording(
                        session_name=payload.get("session_name"),
                        label=payload.get("label"),
                        note=payload.get("note"),
                    )
                    _write_json(self, 200, {"ok": True, "state": state})
                    return
                if path == "/recording/stop":
                    _write_json(self, 200, {"ok": True, "state": manager.stop_recording()})
                    return
                if path == "/recording/mark":
                    state = manager.mark_event(
                        label=str(payload.get("label") or "unknown_noise"),
                        note=payload.get("note"),
                        distance_m=_optional_float(payload.get("distance_m")),
                        bearing_deg=_optional_float(payload.get("bearing_deg")),
                        drone_model=payload.get("drone_model"),
                    )
                    _write_json(self, 200, {"ok": True, "state": state})
                    return
                _write_json(self, 404, {"ok": False, "error": "not_found"})

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    body = handler.rfile.read(length)
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
