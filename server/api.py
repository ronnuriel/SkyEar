from __future__ import annotations
import time
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from shared.event_schema import AcousticEvent, FusedAlert, StationHeartbeat
from server.auth import require_station_auth
from server.database import db
from server.fusion import fuse_events
from server.geo_fusion import map_state_from_db
from server.ptz_dispatcher import dispatch_ptz_for_alert
from server.recording_commands import pending_recording_commands, pop_recording_command, queue_recording_command

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Drone Acoustic Network API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RecordingCommandBody(BaseModel):
    session_name: str | None = None
    label: str | None = None
    note: str | None = None
    distance_m: float | None = None
    bearing_deg: float | None = None
    drone_model: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

def _stamp_receive(payload: AcousticEvent | StationHeartbeat) -> float:
    server_received_unix = time.time()
    payload.server_received_unix = server_received_unix
    metadata = dict(payload.metadata or {})
    metadata["server_received_unix"] = server_received_unix
    station_time = float(payload.timestamp_unix or 0.0)
    if station_time > 0.0 and station_time <= server_received_unix + 300.0:
        metadata["station_to_server_latency_sec"] = server_received_unix - station_time
    payload.metadata = metadata
    return server_received_unix

@app.post("/events")
def ingest_event(event: AcousticEvent, _: None = Depends(require_station_auth)):
    _stamp_receive(event)
    db.add_event(event)
    alert = fuse_events(db.recent_events(limit=200))
    if alert.level > 0:
        db.add_alert(alert)
        dispatch_ptz_for_alert(alert)
    return {
        "ok": True,
        "server_received_unix": event.server_received_unix,
        "alert_level": alert.level,
        "reason": alert.reason,
    }

@app.post("/stations/heartbeat")
def ingest_heartbeat(heartbeat: StationHeartbeat, _: None = Depends(require_station_auth)):
    _stamp_receive(heartbeat)
    db.add_heartbeat(heartbeat)
    command = pop_recording_command(db, heartbeat.station_id)
    return {"ok": True, "server_received_unix": heartbeat.server_received_unix, "recording_command": command}

@app.get("/stations/heartbeat")
def get_latest_heartbeats():
    return {
        station_id: heartbeat.model_dump(mode="json")
        for station_id, heartbeat in db.latest_heartbeats_by_station().items()
    }

@app.get("/stations/health")
def get_station_health():
    return db.station_health()


@app.post("/stations/{station_id}/recording/start")
def recording_start(station_id: str, body: RecordingCommandBody | None = None):
    payload = (body or RecordingCommandBody()).model_dump(exclude_none=True)
    command = queue_recording_command(db, station_id, "start", payload)
    return {"ok": True, "command": command}


@app.post("/stations/{station_id}/recording/stop")
def recording_stop(station_id: str, body: RecordingCommandBody | None = None):
    payload = (body or RecordingCommandBody()).model_dump(exclude_none=True)
    command = queue_recording_command(db, station_id, "stop", payload)
    return {"ok": True, "command": command}


@app.post("/stations/{station_id}/recording/mark")
def recording_mark(station_id: str, body: RecordingCommandBody | None = None):
    payload = (body or RecordingCommandBody()).model_dump(exclude_none=True)
    command = queue_recording_command(db, station_id, "mark", payload)
    return {"ok": True, "command": command}


@app.get("/stations/{station_id}/recording/state")
def recording_state(station_id: str):
    heartbeat = db.latest_heartbeats_by_station().get(station_id)
    metadata = (heartbeat.metadata if heartbeat else {}) or {}
    return {
        "ok": True,
        "station_id": station_id,
        "state": metadata.get("recording_state") or {},
        "last_command_result": metadata.get("recording_command_result"),
        "pending_commands": pending_recording_commands(db, station_id),
    }

@app.get("/events")
def get_events(limit: int = 100):
    return [e.model_dump(mode="json") for e in db.recent_events(limit=limit)]

@app.get("/alerts")
def get_alerts(limit: int = 50):
    return [a.model_dump(mode="json") for a in db.recent_alerts(limit=limit)]

@app.get("/fusion")
def get_fusion():
    alert: FusedAlert = fuse_events(db.recent_events(limit=200))
    return alert.model_dump(mode="json")

@app.get("/map/state")
def get_map_state():
    return map_state_from_db(db)


def _latest_by_station_json() -> dict:
    return {
        station_id: event.model_dump(mode="json")
        for station_id, event in db.latest_by_station().items()
    }


def _station_health_summary(stations_health: dict) -> list[dict]:
    rows = []
    for station_id, health in sorted(stations_health.items()):
        heartbeat = health.get("heartbeat") or {}
        event = health.get("latest_event") or {}
        metadata = (event.get("metadata") or {}) or (heartbeat.get("metadata") or {})
        rows.append(
            {
                "station_id": station_id,
                "station_name": health.get("station_name"),
                "alive_state": health.get("alive_state"),
                "health": "degraded" if health.get("alive_state") == "error" else health.get("alive_state"),
                "last_status": health.get("last_event_status"),
                "heartbeat_age_sec": health.get("heartbeat_age_sec"),
                "event_age_sec": health.get("event_age_sec"),
                "latency_sec": health.get("latency_sec"),
                "latitude": metadata.get("latitude") or (event.get("station_location") or {}).get("latitude"),
                "longitude": metadata.get("longitude") or (event.get("station_location") or {}).get("longitude"),
                "line_id": metadata.get("line_id"),
                "line_distance_m": metadata.get("line_distance_m"),
                "fiber_node_id": metadata.get("fiber_node_id"),
                "fiber_connected": metadata.get("fiber_connected"),
                "station_mode": event.get("station_mode") or heartbeat.get("station_mode"),
                "simulation": (
                    str(metadata.get("source") or "").startswith("simulate_")
                    or str(event.get("station_mode") or heartbeat.get("station_mode") or "") == "simulation"
                ),
            }
        )
    return rows


@app.get("/dashboard/state")
def get_dashboard_state(alert_limit: int = 50):
    now = time.time()
    events = db.recent_events(limit=200)
    fusion = fuse_events(events)
    stations_health = db.station_health(now=now)
    return {
        "server_time": now,
        "health": health(),
        "fusion": fusion.model_dump(mode="json"),
        "map_state": map_state_from_db(db, now=now),
        "stations_latest": _latest_by_station_json(),
        "stations_health": stations_health,
        "alerts": [a.model_dump(mode="json") for a in db.recent_alerts(limit=alert_limit)],
    }


@app.get("/dashboard/live")
def get_dashboard_live(bearing_cues: bool = False):
    now = time.time()
    events = db.recent_events(limit=200)
    fusion = fuse_events(events).model_dump(mode="json")
    map_state = map_state_from_db(db, now=now)
    stations_health = db.station_health(now=now)
    return {
        "server_time": now,
        "fusion": {
            "timestamp_unix": fusion.get("timestamp_unix"),
            "level": fusion.get("level"),
            "global_level": fusion.get("global_level"),
            "status": fusion.get("status"),
            "confidence": fusion.get("confidence"),
            "interpretation": fusion.get("interpretation"),
            "reason": fusion.get("reason"),
            "tracks": fusion.get("tracks") or [],
        },
        "map_state": {
            "server_time": map_state.get("server_time"),
            "stations": map_state.get("stations") or [],
            "tracks": map_state.get("tracks") or [],
            "track_geo_estimates": map_state.get("track_geo_estimates") or [],
            "geo_estimate_suppressed_reason": map_state.get("geo_estimate_suppressed_reason"),
            "geo_estimates": [],
            "bearing_cues": (map_state.get("bearing_cues") or []) if bearing_cues else [],
        },
        "stations_health_summary": _station_health_summary(stations_health),
    }


@app.get("/live")
def get_live_page():
    return FileResponse(STATIC_DIR / "live.html")

@app.get("/stations/latest")
def get_latest_by_station():
    return _latest_by_station_json()

@app.get("/stations/summary")
def get_station_summary():
    return [
        {
            "station_id": event.station_id,
            "station_name": event.station_name,
            "status": event.status.value,
            "confidence": event.confidence,
            "harmonic_score": event.harmonic_score,
            "best_f0_hz": event.best_f0_hz,
            "channel_agreement_count": event.channel_agreement_count,
            "channel_count": event.channel_count,
            "strongest_channel": event.strongest_channel,
            "calibrated": event.calibrated,
            "timestamp_unix": event.timestamp_unix,
            "server_received_unix": event.server_received_unix,
        }
        for event in db.latest_by_station().values()
    ]
