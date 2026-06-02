from shared.event_schema import AcousticEvent, FusedAlert
from server.alert_logic import alert_level_from_recent_events

def fuse_events(events: list[AcousticEvent]) -> FusedAlert:
    return alert_level_from_recent_events(events)
