from __future__ import annotations
import time
from shared.event_schema import AcousticEvent, FusedAlert

def alert_level_from_recent_events(events: list[AcousticEvent], window_sec: float = 8.0) -> FusedAlert:
    now = time.time()
    recent = [e for e in events if now - e.timestamp_unix <= window_sec]
    suspect_or_more = [e for e in recent if e.status in {"suspect", "drone_like", "alert"}]
    alerts = [e for e in recent if e.status == "alert"]

    suspect_stations = {e.station_id for e in suspect_or_more}
    alert_stations = {e.station_id for e in alerts}

    if len(alert_stations) >= 2:
        level, reason = 3, "Two or more stations reported alert-level rotor evidence."
    elif len(alerts) >= 1 or len(suspect_stations) >= 2:
        level, reason = 2, "One alert station or multiple suspect stations."
    elif len(suspect_or_more) >= 1:
        level, reason = 1, "One station reported suspect rotor evidence."
    else:
        level, reason = 0, "No recent acoustic evidence."

    confidence = max([float(e.confidence) for e in suspect_or_more], default=0.0)
    if len(suspect_stations) >= 2:
        confidence = min(1.0, confidence + 0.15)

    return FusedAlert(
        timestamp_unix=now,
        level=level,
        status=f"LEVEL_{level}",
        confidence=confidence,
        reason=reason,
        events_used=suspect_or_more[-10:],
    )
