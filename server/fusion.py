from shared.event_schema import AcousticEvent, FusedAlert
from server.track_fusion import fuse_tracks

def fuse_events(events: list[AcousticEvent]) -> FusedAlert:
    return fuse_tracks(events)
