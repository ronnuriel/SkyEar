from __future__ import annotations
from fastapi import FastAPI
from shared.event_schema import AcousticEvent, FusedAlert
from server.database import db
from server.fusion import fuse_events
from server.ptz_dispatcher import dispatch_ptz_for_alert

app = FastAPI(title="Drone Acoustic Network API")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/events")
def ingest_event(event: AcousticEvent):
    db.add_event(event)
    alert = fuse_events(db.recent_events(limit=200))
    if alert.level > 0:
        db.add_alert(alert)
        dispatch_ptz_for_alert(alert)
    return {"ok": True, "alert_level": alert.level, "reason": alert.reason}

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
