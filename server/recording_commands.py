from __future__ import annotations

from typing import Any


def queue_recording_command(db: Any, station_id: str, action: str, payload: dict | None = None) -> dict:
    return db.queue_recording_command(station_id, action, payload or {})


def pop_recording_command(db: Any, station_id: str) -> dict | None:
    return db.pop_recording_command(station_id)


def pending_recording_commands(db: Any, station_id: str) -> list[dict]:
    return db.pending_recording_commands(station_id)
