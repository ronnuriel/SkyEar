from __future__ import annotations
from collections import deque
import time

from shared.event_schema import AcousticEvent, FusedAlert, StationHeartbeat

class InMemoryDatabase:
    def __init__(self, max_events: int = 2000, max_alerts: int = 500, max_heartbeats: int = 2000):
        self.events = deque(maxlen=max_events)
        self.alerts = deque(maxlen=max_alerts)
        self.heartbeats = deque(maxlen=max_heartbeats)

    def add_event(self, event: AcousticEvent):
        self.events.append(event)

    def add_alert(self, alert: FusedAlert):
        self.alerts.append(alert)

    def add_heartbeat(self, heartbeat: StationHeartbeat):
        self.heartbeats.append(heartbeat)

    def recent_events(self, limit: int = 100):
        return list(self.events)[-limit:]

    def recent_alerts(self, limit: int = 100):
        return list(self.alerts)[-limit:]

    def latest_by_station(self) -> dict[str, AcousticEvent]:
        latest: dict[str, AcousticEvent] = {}
        for event in self.events:
            latest[event.station_id] = event
        return latest

    def latest_heartbeats_by_station(self) -> dict[str, StationHeartbeat]:
        latest: dict[str, StationHeartbeat] = {}
        for heartbeat in self.heartbeats:
            latest[heartbeat.station_id] = heartbeat
        return latest

    def station_health(self, now: float | None = None) -> dict[str, dict]:
        now = time.time() if now is None else float(now)
        latest_events = self.latest_by_station()
        latest_heartbeats = self.latest_heartbeats_by_station()
        station_ids = sorted(set(latest_events) | set(latest_heartbeats))
        health: dict[str, dict] = {}

        for station_id in station_ids:
            heartbeat = latest_heartbeats.get(station_id)
            event = latest_events.get(station_id)
            heartbeat_received = heartbeat.server_received_unix if heartbeat else None
            event_received = event.server_received_unix if event else None
            if event_received is None and event is not None:
                event_received = float((event.metadata or {}).get("server_received_unix") or event.timestamp_unix)

            heartbeat_age = None if heartbeat_received is None else max(0.0, now - float(heartbeat_received))
            event_age = None if event_received is None else max(0.0, now - float(event_received))
            if heartbeat_age is None:
                alive_state = "offline"
            elif heartbeat.status == "error":
                alive_state = "error"
            elif heartbeat_age <= 10.0:
                alive_state = "online"
            elif heartbeat_age <= 30.0:
                alive_state = "stale"
            else:
                alive_state = "offline"

            latency = None
            if heartbeat is not None:
                latency = (heartbeat.metadata or {}).get("station_to_server_latency_sec")
            if latency is None and event is not None:
                latency = (event.metadata or {}).get("station_to_server_latency_sec")

            health[station_id] = {
                "station_id": station_id,
                "station_name": (heartbeat.station_name if heartbeat else None) or (event.station_name if event else None),
                "alive_state": alive_state,
                "heartbeat_age_sec": heartbeat_age,
                "event_age_sec": event_age,
                "timestamp_unix": heartbeat.timestamp_unix if heartbeat else None,
                "server_received_unix": heartbeat.server_received_unix if heartbeat else None,
                "last_event_timestamp_unix": event.timestamp_unix if event else None,
                "last_event_server_received_unix": event.server_received_unix if event else None,
                "last_event_status": (event.status.value if hasattr(event.status, "value") else event.status)
                if event
                else (heartbeat.last_event_status if heartbeat else None),
                "last_error": heartbeat.last_error if heartbeat else None,
                "audio_device": heartbeat.audio_device if heartbeat else None,
                "sample_rate": heartbeat.sample_rate if heartbeat else None,
                "channels": heartbeat.channels if heartbeat else None,
                "calibrated": heartbeat.calibrated if heartbeat else (event.calibrated if event else None),
                "latency_sec": None if latency is None else float(latency),
                "heartbeat": heartbeat.model_dump(mode="json") if heartbeat else None,
                "latest_event": event.model_dump(mode="json") if event else None,
            }
        return health

db = InMemoryDatabase()
