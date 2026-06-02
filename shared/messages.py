from dataclasses import dataclass

@dataclass(frozen=True)
class Topics:
    acoustic_events: str = "drone/acoustic/events"
    fused_alerts: str = "drone/fusion/alerts"
    station_status: str = "drone/stations/status"
