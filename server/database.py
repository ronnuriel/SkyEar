from __future__ import annotations
from collections import deque
from shared.event_schema import AcousticEvent, FusedAlert

class InMemoryDatabase:
    def __init__(self, max_events: int = 2000, max_alerts: int = 500):
        self.events = deque(maxlen=max_events)
        self.alerts = deque(maxlen=max_alerts)

    def add_event(self, event: AcousticEvent):
        self.events.append(event)

    def add_alert(self, alert: FusedAlert):
        self.alerts.append(alert)

    def recent_events(self, limit: int = 100):
        return list(self.events)[-limit:]

    def recent_alerts(self, limit: int = 100):
        return list(self.alerts)[-limit:]

    def latest_by_station(self) -> dict[str, AcousticEvent]:
        latest: dict[str, AcousticEvent] = {}
        for event in self.events:
            latest[event.station_id] = event
        return latest

db = InMemoryDatabase()
