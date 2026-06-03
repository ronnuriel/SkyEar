from shared.event_schema import FusedAlert
from station.ptz_client import PTZClient

def dispatch_ptz_for_alert(alert: FusedAlert):
    # Camera-only visual confirmation placeholder.
    if alert.level >= 2:
        print("[PTZ] candidate alert; dispatch camera for visual confirmation only")
        if alert.tracks:
            first_track = alert.tracks[0]
            first_event = first_track.events[0] if first_track.events else None
            bearing = first_event.estimated_azimuth_deg if first_event is not None else None
            PTZClient().slew_to_cue(bearing, track_id=first_track.track_id)
